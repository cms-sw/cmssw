// $Id: CurlInterface.h,v 1.1 2009/08/20 13:43:18 mommsen Exp $
/// @file: CurlInterface.h

#ifndef StorageManager_CurlInterface_h
#define StorageManager_CurlInterface_h

#include <string>
#include <curl/curl.h>


namespace stor {

  /**
   * Helper class to interact with curl
   *
   * $Author: mommsen $
   * $Revision: 1.1 $
   * $Date: 2009/08/20 13:43:18 $
   */
 
  class CurlInterface
  {

  public:

    /**
     * Get webpage content from specified URL using the user/password
     * specified.
     * If the return value is CURLE_OK, the webpage could be fetched
     * and the content is in the content string. Otherwise, the 
     * content string contains the error message.
     */
    CURLcode getContent(const std::string& url, const std::string& user, std::string& content);
    
    
  private:
    
    static size_t writeToString(char* data, size_t size, size_t nmemb, std::string* buffer);
    
    char errorBuffer[CURL_ERROR_SIZE]; 
  };

} // namespace stor

#endif // StorageManager_CurlInterface_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
