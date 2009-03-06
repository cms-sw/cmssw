
#include "EventFilter/Utilities/interface/SquidNet.h"
#include "EventFilter/Utilities/interface/CurlUtils.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "curl/curl.h"
#include <netdb.h>
#include <sys/socket.h>        /* for AF_INET */

#include <fstream>


namespace evf{
  
  SquidNet::SquidNet(unsigned int proxyport, std::string const &url) : port_(proxyport), urlToGet_(url) 
  {
    std::ostringstream oproxy;
    oproxy << "localhost:" << port_;
    proxy_ = oproxy.str();
  }
  bool SquidNet::check(){
    bool retVal = true;
    
    CURL* han = curl_easy_init();
    if(han==0)
      {
	XCEPT_RAISE(evf::Exception,"could not create handle for SquidNet fisher");
      }
    char error[CURL_ERROR_SIZE];
    std::string dummy;

    struct curl_slist *headers=NULL; /* init to NULL is important */

    headers = curl_slist_append(headers, "Pragma:");

    curl_easy_setopt(han, CURLOPT_HTTPHEADER, headers);
 
    curl_easy_setopt(han, CURLOPT_PROXY, proxy_.c_str());
    
    curl_easy_setopt(han, CURLOPT_URL, urlToGet_.c_str());
    
    curl_easy_setopt(han, CURLOPT_WRITEFUNCTION, &write_data);
    curl_easy_setopt(han, CURLOPT_WRITEDATA, &dummy);
    curl_easy_setopt(han, CURLOPT_ERRORBUFFER, error);
    int success = curl_easy_perform(han);

    curl_slist_free_all(headers); /* free the header list */

    curl_easy_cleanup(han);
    if(success != 0)
      retVal = false;
    return retVal;
  }

}
