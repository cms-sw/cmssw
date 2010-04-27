#ifndef CURLPOSTER_H
#define CURLPOSTER_H

#include <sys/utsname.h>
#include <string>

namespace evf {


  class CurlPoster
  {
  public:
    CurlPoster(const std::string& url) : url_(url), active_(true){
      buf_=(struct utsname*)new char[sizeof(struct utsname)];
      uname(buf_);
      //      check();
    }
    virtual ~CurlPoster(){delete [] buf_;}
    bool check(int);
    void postBinary(const char *, unsigned int, unsigned int); 
    void postString(const char *, size_t, unsigned int); 
  private:
    enum mode{text,bin};  
    void post(const char *, unsigned int, unsigned int, mode);
    std::string url_;
    bool active_;
    struct utsname* buf_; 
  };

} // evf

#endif
