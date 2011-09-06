// $Id: CurlInterface.cc,v 1.2.6.1 2011/03/07 11:33:04 mommsen Exp $
/// @file: CurlInterface.cc

#include "EventFilter/StorageManager/interface/CurlInterface.h"

using namespace stor;


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

  CURLcode status = do_curl(curl, url, content);
  curl_slist_free_all(headers);

  return status;
}


CURLcode CurlInterface::do_curl(CURL* curl, const std::string& url, Content& content)
{
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 4); // seconds
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1); // do not send any signals
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stor::CurlInterface::writeToString);  
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &content);  
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuffer_); 
  curl_easy_setopt(curl, CURLOPT_FAILONERROR,1); 
  
  CURLcode returnCode = curl_easy_perform(curl);
  
  curl_easy_cleanup(curl);
  
  if (returnCode != CURLE_OK)
  {
    size_t i = 0;
    content.clear();
    while ( errorBuffer_[i] != '\0' )
    {
      content.push_back( errorBuffer_[i] );
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


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
