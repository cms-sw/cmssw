// $Id: CurlInterface.cc,v 1.4 2011/11/16 14:32:22 mommsen Exp $
/// @file: CurlInterface.cc

#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/interface/Exception.h"

using namespace stor;

CurlInterfacePtr CurlInterface::interface_;
pthread_mutex_t* CurlInterface::mutexes_ = 0;


CurlInterface::CurlInterface()
{
  curl_global_init(CURL_GLOBAL_ALL);
  const int cryptoNumLocks = CRYPTO_num_locks();

  //setup array to store all of the mutexes available to OpenSSL.
  mutexes_ = (pthread_mutex_t*)malloc( cryptoNumLocks * sizeof(pthread_mutex_t) );
  if ( ! mutexes_ )
    XCEPT_RAISE(stor::exception::Exception, "Failed to allocate memory for SSL mutexes");

  for (int i = 0; i < cryptoNumLocks; ++i)
    pthread_mutex_init(&mutexes_[i],0);

  CRYPTO_set_id_callback(sslIdFunction);
  CRYPTO_set_locking_callback(sslLockingFunction);
}


CurlInterface::~CurlInterface()
{
  CRYPTO_set_id_callback(0);
  CRYPTO_set_locking_callback(0);

  for (int i = 0; i < CRYPTO_num_locks(); ++i)
    pthread_mutex_destroy(&mutexes_[i]);
  free(mutexes_);
  mutexes_ = NULL;
  
  curl_global_cleanup();
}


boost::shared_ptr<CurlInterface> CurlInterface::getInterface()
{
  if (interface_.get() == 0)
  {
    interface_.reset( new CurlInterface() );
  }

  return interface_;
}


CURLcode CurlInterface::getContent
(
  const std::string& url,
  const std::string& user,
  Content& content
)
{
  CURL* curl = curl_easy_init();
  if ( ! curl ) return CURLE_FAILED_INIT;

  curl_easy_setopt(curl, CURLOPT_USERPWD, user.c_str());

  return do_curl(curl, url, content);
}


CURLcode CurlInterface::postBinaryMessage
(
  const std::string& url,
  void* buf,
  size_t size,
  Content& content
)
{
  CURL* curl = curl_easy_init();
  if ( ! curl ) return CURLE_FAILED_INIT;

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buf);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, size);

  struct curl_slist *headers=NULL;
  headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
  headers = curl_slist_append(headers, "Content-Transfer-Encoding: binary");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  const CURLcode status = do_curl(curl, url, content);
  curl_slist_free_all(headers);

  return status;
}


CURLcode CurlInterface::do_curl(CURL* curl, const std::string& url, Content& content)
{
  char errorBuffer[CURL_ERROR_SIZE]; 

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 4); // seconds
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1); // do not send any signals
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stor::CurlInterface::writeToString);  
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &content);  
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuffer); 
  curl_easy_setopt(curl, CURLOPT_FAILONERROR,1); 
  
  const CURLcode returnCode = curl_easy_perform(curl);
  
  curl_easy_cleanup(curl);
  
  if (returnCode != CURLE_OK)
  {
    size_t i = 0;
    content.clear();
    while ( errorBuffer[i] != '\0' )
    {
      content.push_back( errorBuffer[i] );
      ++i;
    }
    content.push_back('\0');
  }

  return returnCode;
}


size_t CurlInterface::writeToString(char *data, size_t size, size_t nmemb, Content* buffer)
{
  if (buffer == NULL) return 0;

  const size_t length = size * nmemb;
  buffer->insert(buffer->end(), data, data+length);
  return length;
}


void CurlInterface::sslLockingFunction(int mode, int n, const char* file, int line)
{
  if (mode & CRYPTO_LOCK)
    pthread_mutex_lock(&mutexes_[n]);
  else
    pthread_mutex_unlock(&mutexes_[n]);
}


unsigned long CurlInterface::sslIdFunction(void)
{
  return ( (unsigned long)pthread_self() );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
