#ifndef SMCurlInterface_H
#define SMCurlInterface_H

/**
 *  This will eventually be an interface class for curl common
 *  functions but now is just some common utility
 *
 *  $Id: SMCurlInterface.h,v 1.2 2009/05/09 01:28:22 elmer Exp $
 */

#include <string>
#include <iostream>
#include <cstdlib>

namespace stor
{
  struct ReadData
  {
    std::string d_;
  };  

  size_t func(void* buf,size_t size, size_t nmemb, void* userp);

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
