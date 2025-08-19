import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain.schema import Document  # ✅ Added missing import
from pytube import YouTube
from urllib.parse import urlparse, parse_qs
import re
import requests
from typing import Optional, List
## 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  mm

# Page setup
st.set_page_config(
    page_title="YouTube & Web Summarizer",
    page_icon="📺",
    layout="wide"
)

st.title("📺 YouTube & Web Content Summarizer")
st.markdown("Get concise summaries of YouTube videos or web articles")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    st.markdown("[Get Groq API Key](https://console.groq.com/keys)")
    st.markdown("### YouTube Tips:")
    st.markdown("- Videos must have English captions")
    st.markdown("- Doesn't work with age-restricted/private videos")

# URL input
url = st.text_input(
    "Enter YouTube or website URL",
    placeholder="https://www.youtube.com/watch?v=... or https://example.com",
    help="For YouTube, ensure captions are available"
)

# Enhanced headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def is_valid_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL with video ID"""
    parsed = urlparse(url)
    domains = ('youtube.com', 'www.youtube.com', 'youtu.be')
    if parsed.netloc not in domains:
        return False
    return extract_youtube_id(url) is not None

def load_youtube_transcript(video_id: str):
    """Improved YouTube transcript loading with better error handling"""
    try:
        # First try with standard loader
        loader = YoutubeLoader(
            video_id=video_id,
            add_video_info=True,
            language=["en"],
            continue_on_failure=False
        )
        docs = loader.load()
        
        if docs and len(docs[0].page_content) > 50:  # Minimum content check
            return docs
            
        # Fallback to pytube for manual caption extraction
        yt = YouTube(f"https://youtu.be/{video_id}")
        if yt.captions:
            caption = yt.captions.get_by_language_code('en') or yt.captions.all()[0]
            if caption:
                return [Document(
                    page_content=caption.generate_srt_captions(),
                    metadata={"title": yt.title, "author": yt.author}
                )]
                
        raise Exception("No English captions available")
        
    except Exception as e:
        # Check for age restriction
        yt = YouTube(f"https://youtu.be/{video_id}")
        if yt.age_restricted:
            raise Exception("Age-restricted video - cannot access automatically")
        raise

def test_url_accessibility(url: str) -> bool:
    """Check if URL is accessible"""
    if is_valid_youtube_url(url):
        return True  # Skip check for YouTube
    try:
        response = requests.head(url, headers=HEADERS, timeout=10)
        return response.status_code < 400
    except:
        return False

# Prompt template
prompt_template = """Create a concise summary of the following content in 250-300 words:

{text}

SUMMARY:"""
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

# Processing
if st.button("Generate Summary", type="primary"):
    if not groq_api_key.strip():
        st.error("Please enter your Groq API Key")
    elif not url.strip():
        st.error("Please enter a URL")
    elif not validators.url(url):
        st.error("Invalid URL format")
    elif not test_url_accessibility(url):
        st.error("URL not accessible - check if it's correct and public")
    else:
        try:
            with st.spinner("Processing content..."):
                documents = None
                
                # Handle YouTube URLs
                if is_valid_youtube_url(url):
                    video_id = extract_youtube_id(url)
                    st.info(f"Processing YouTube video ID: {video_id}")
                    
                    try:
                        documents = load_youtube_transcript(video_id)
                        if not documents:
                            raise Exception("No transcript available")
                    except Exception as e:
                        st.error(f"Failed to process YouTube video: {str(e)}")
                        st.markdown("""
                        **Common YouTube issues:**
                        - Video has no English captions
                        - Age-restricted content
                        - Private/unlisted video
                        - YouTube API limitations
                        """)
                        st.stop()
                
                # Handle regular URLs
                else:
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[url],
                            ssl_verify=True,
                            headers=HEADERS,
                            strategy="fast"
                        )
                        documents = loader.load()
                    except Exception as e:
                        st.error(f"Failed to load webpage: {str(e)}")
                        st.stop()

                # Initialize LLM and generate summary
                llm = ChatGroq(
                    model_name="Llama3-8b-8192",
                    groq_api_key=groq_api_key,
                    temperature=0.3,
                    max_tokens=1024
                )

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt,
                    verbose=False
                )

                result = chain.invoke({"input_documents": documents})
                
                # Display results
                st.success("✅ Summary generated successfully!")
                st.subheader("Summary")
                st.write(result["output_text"])

                # Show metadata if available
                if documents and documents[0].metadata:
                    st.divider()
                    st.subheader("Source Information")
                    if "title" in documents[0].metadata:
                        st.write(f"**Title:** {documents[0].metadata['title']}")
                    if "author" in documents[0].metadata:
                        st.write(f"**Author:** {documents[0].metadata['author']}")

        except Exception as e:

            st.error(f"Processing error: {str(e)}")
