#include "EventFilter/Utilities/interface/CurlPoster.h"
#include "EventFilter/Utilities/interface/CurlUtils.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "curl/curl.h"
#include <netdb.h>
#include <sys/socket.h>        /* for AF_INET */

#include <sstream>



namespace evf{
  //______________________________________________________________________________
  void CurlPoster::post(unsigned char *content, unsigned int len, unsigned int run, mode m)
  {
    std::string urlp = url_+"/postEntry";
    char srun[12];
    sprintf(srun,"%d",run);
    CURL* han = curl_easy_init();
    if(han==0)
      {
	XCEPT_RAISE(evf::Exception,"could not create handle for curlPoster"); 
      }
    struct curl_slist *headers=NULL; /* init to NULL is important */
    switch(m){
    case text:
      {
	headers = curl_slist_append(headers, "Content-Type: text/plain");
	break;
      }
    case bin:
      {
	headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
	headers = curl_slist_append(headers, "Content-Transfer-Encoding: base64");
	break;
      }
    default:
      {
	headers = curl_slist_append(headers, "Content-Type: application/xml");
      }
    }
    struct curl_httppost *post=NULL;
    struct curl_httppost *last=NULL;
    char error[CURL_ERROR_SIZE];
    
    curl_easy_setopt(han, CURLOPT_URL, urlp.c_str());
    curl_easy_setopt(han, CURLOPT_VERBOSE);
    curl_easy_setopt(han, CURLOPT_NOSIGNAL);
    //	curl_easy_setopt(han, CURLOPT_TIMEOUT, 60.0L);
    curl_formadd(&post, &last,
		 CURLFORM_COPYNAME, "name",
		 CURLFORM_COPYCONTENTS, buf_->nodename, CURLFORM_END);
    curl_formadd(&post, &last,
		 CURLFORM_COPYNAME, "run",
		 CURLFORM_COPYCONTENTS, srun, CURLFORM_END);
    curl_formadd(&post, &last,
		 CURLFORM_COPYNAME, "trp",
		 CURLFORM_COPYCONTENTS, content,
		 CURLFORM_CONTENTSLENGTH, len,
		 CURLFORM_CONTENTHEADER, headers,
		 CURLFORM_END);
    curl_easy_setopt(han, CURLOPT_HTTPPOST, post);
    curl_easy_setopt(han, CURLOPT_ERRORBUFFER, error);
	
    int success = curl_easy_perform(han);
    curl_formfree(post);
    curl_slist_free_all(headers); /* free the header list */
    curl_easy_cleanup(han);
    
    if(success != 0)
      {
	std::ostringstream msg;
	msg <<  "could not post data to url " << url_ << " error #" 
	    << success << " " << error;
	XCEPT_RAISE(evf::Exception,msg.str().c_str());
      }

  }
  void CurlPoster::postString(unsigned char *content, unsigned int len, unsigned int run)
  {
    if(!active_) return;
    std::cout << "==============doing postString " << std::endl;
    post(content,len,run,text)
  }
  void CurlPoster::postBinary(unsigned char *content, unsigned int len, unsigned int run)
  {
    if(!active_) return;
    std::cout << "==============doing postBinary " << std::endl;
    post(content,len,run,bin);
  }

  bool CurlPoster::check()
  {
    bool retVal = true;
    
    CURL* han = curl_easy_init();
    if(han==0)
      {
	active_ = false;
      }
    char error[CURL_ERROR_SIZE];
    std::string dummy;

    struct curl_slist *headers=NULL; /* init to NULL is important */

    headers = curl_slist_append(headers, "Pragma:");

    curl_easy_setopt(han, CURLOPT_HTTPHEADER, headers);
 
    curl_easy_setopt(han, CURLOPT_URL, url_.c_str());
    
    curl_easy_setopt(han, CURLOPT_WRITEFUNCTION, &write_data);
    curl_easy_setopt(han, CURLOPT_WRITEDATA, &dummy);
    curl_easy_setopt(han, CURLOPT_ERRORBUFFER, error);
    int success = curl_easy_perform(han);

    curl_slist_free_all(headers); /* free the header list */

    curl_easy_cleanup(han);
    if(success != 0){
      retVal = false;
      active_ = false;
    }
    return retVal;

  }

} //end namespace evf
