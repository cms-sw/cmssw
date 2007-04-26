#ifndef SMCurlInterface_H
#define SMCurlInterface_H

/** 
 *  This will eventually be an interface class for curl common
 *  functions but now is just some common utility
 *
 *  $Id$
 */

#include <string>
#include <iostream>

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
