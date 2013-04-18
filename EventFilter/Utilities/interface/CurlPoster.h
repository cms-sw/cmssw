#ifndef CURLPOSTER_H
#define CURLPOSTER_H

#include <sys/utsname.h>
#include <string>

namespace evf {


  class CurlPoster
  {
  public:

    enum mode{text,stack,leg,bin};  

    CurlPoster(const std::string& url) : url_(url), active_(true){
      buf_=(struct utsname*)new char[sizeof(struct utsname)];
      uname(buf_);
      //      check();
    }
    virtual ~CurlPoster(){delete [] buf_;}
    bool check(int);
    void postBinary(const unsigned char *, size_t, unsigned int
		    , const std::string& = standard_post_method_); 
    void postString(const char *, size_t, unsigned int 
		    , mode, const std::string& = standard_post_method_); 
  private:
    void post(const unsigned char *, size_t, unsigned int, mode, const std::string&);
    std::string url_;
    bool active_;
    struct utsname* buf_; 
    static const std::string standard_post_method_;
  };

} // evf

#endif
