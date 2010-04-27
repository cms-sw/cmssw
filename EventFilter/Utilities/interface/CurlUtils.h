#ifndef EVF_CURLUTILS_H
#define EVF_CURLUTILS_H

#include <string>
#include <sstream>
#include <cstring>
#include <cstdio>

namespace evf{

  //______________________________________________________________________________
  static size_t write_data(void *ptr, size_t size, size_t nmemb, void *pointer)
  {
    using std::string;
    using std::ostringstream; 
    char *cfg = new char[size*nmemb+1];
    string *spt = (string *)pointer;
    strncpy(cfg,(const char*)ptr,size*nmemb);
    cfg[size*nmemb] = '\0';
    ostringstream output;
    output<<cfg;
    delete[] cfg;
    (*spt) += output.str(); 
    return size*nmemb;
  }

}

#endif
