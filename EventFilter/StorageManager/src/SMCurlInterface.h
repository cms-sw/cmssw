#ifndef SMCurlInterface_H
#define SMCurlInterface_H

/** 
 *  This will eventually be an interface class for curl common
 *  functions but now is just some common utility
 *
 *  $Id: SMCurlInterface.h,v 1.1 2007/04/04 22:14:27 hcheung Exp $
 */

namespace stor
{
  struct ReadData
  {
    std::string d_;
  };  

  size_t func(void* buf,size_t size, size_t nmemb, void* userp)
  {
    ReadData* rdata = (ReadData*)userp;
    size_t sz = size * nmemb;
    char* cbuf = (char*)buf;
    rdata->d_.insert(rdata->d_.end(),cbuf,cbuf+sz);
    return sz;
  }

  template <class Han, class Opt, class Par>
  int setopt(Han han,Opt opt,Par par)
  {
    if(curl_easy_setopt(han,opt,par)!=0)
      {
        std::cerr << "could not stor::setopt " << opt << std::endl;
        abort();
      }
    return 0;
  }
}
#endif
