// $Id: CurlInterface.cc,v 1.1 2009/08/20 13:44:38 mommsen Exp $
/// @file: CurlInterface.cc

#include "EventFilter/StorageManager/interface/CurlInterface.h"

using namespace stor;


CURLcode CurlInterface::getContent
(
  const std::string& url,
  const std::string& user,
  std::string& content
)
{
  //  std::string buffer;

  CURL* curl = curl_easy_init();
  if ( ! curl ) return CURLE_FAILED_INIT;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERPWD, user.c_str());
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 4); // seconds
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1); // do not send any signals
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stor::CurlInterface::writeToString);  
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &content);  
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuffer); 
  curl_easy_setopt(curl, CURLOPT_FAILONERROR,1); 
  
  CURLcode returnCode = curl_easy_perform(curl);
  
  curl_easy_cleanup(curl);
  
  if (returnCode != CURLE_OK)
    content = errorBuffer;
  
  return returnCode;
}


size_t CurlInterface::writeToString(char *data, size_t size, size_t nmemb, std::string *buffer)
{
  int result = 0;

  if (buffer != NULL)
  {
    buffer->append(data, size * nmemb);
    result = size * nmemb;
  }
  
  return result;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
