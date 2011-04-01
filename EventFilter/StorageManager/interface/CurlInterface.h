// $Id: CurlInterface.h,v 1.2.4.3 2011/02/28 17:56:15 mommsen Exp $
/// @file: CurlInterface.h

#ifndef EventFilter_StorageManager_CurlInterface_h
#define EventFilter_StorageManager_CurlInterface_h

#include <curl/curl.h>
#include <string>
#include <vector>


namespace stor {

  /**
   * Helper class to interact with curl
   *
   * $Author: mommsen $
   * $Revision: 1.2.4.3 $
   * $Date: 2011/02/28 17:56:15 $
   */
 
  class CurlInterface
  {

  public:

    typedef std::vector<char> Content;

    /**
     * Get webpage content from specified URL using the user/password
     * specified.
     * If the return value is CURLE_OK, the webpage could be fetched
     * and the content is in the content string. Otherwise, the 
     * content string contains the error message.
     */
    CURLcode getContent(const std::string& url, const std::string& user, Content& content);

    /**
     * Post message a message at the given location.
     * If the return value is CURLE_OK, the post succeeded
     * and the reply is in the content string. Otherwise, the 
     * content string contains the error message.
     */
    CURLcode postBinaryMessage(const std::string& url, void* buf, size_t size, Content& content);

    
  private:

    CURLcode do_curl(CURL*, const std::string& url, Content& content);
    static size_t writeToString(char* data, size_t size, size_t nmemb, Content* buffer);
    
    char errorBuffer_[CURL_ERROR_SIZE]; 
  };

} // namespace stor

#endif // EventFilter_StorageManager_CurlInterface_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
