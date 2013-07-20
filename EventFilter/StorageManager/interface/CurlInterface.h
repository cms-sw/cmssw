// $Id: CurlInterface.h,v 1.4 2011/11/16 14:32:22 mommsen Exp $
/// @file: CurlInterface.h

#ifndef EventFilter_StorageManager_CurlInterface_h
#define EventFilter_StorageManager_CurlInterface_h

#include "boost/shared_ptr.hpp"

#include <curl/curl.h>
#include <openssl/crypto.h>
#include <pthread.h>
#include <string>
#include <vector>


namespace stor {

  /**
   * Helper class to interact with curl
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2011/11/16 14:32:22 $
   */
 
  class CurlInterface
  {

  public:

    typedef std::vector<char> Content;

    ~CurlInterface();

    /**
     * Return a shared pointer to the singleton
     */
    static boost::shared_ptr<CurlInterface> getInterface();

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

    CurlInterface();

    CURLcode do_curl(CURL*, const std::string& url, Content& content);
    static size_t writeToString(char* data, size_t size, size_t nmemb, Content* buffer);
    static void sslLockingFunction(int mode, int n, const char* file, int line);
    static unsigned long sslIdFunction();
    
    static boost::shared_ptr<CurlInterface> interface_;
    static pthread_mutex_t* mutexes_;
  };

  typedef boost::shared_ptr<CurlInterface> CurlInterfacePtr;

} // namespace stor

#endif // EventFilter_StorageManager_CurlInterface_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
