// $Id: CurlInterface.h,v 1.2.6.1 2011/03/07 11:33:04 mommsen Exp $
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
   * $Revision: 1.2.6.1 $
   * $Date: 2011/03/07 11:33:04 $
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
