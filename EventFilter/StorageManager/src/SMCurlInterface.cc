/** 
 *  This will eventually be an interface class for curl common
 *  functions but now is just some common utility
 *
 *  $Id$
 */

#include "EventFilter/StorageManager/interface/SMCurlInterface.h"

namespace stor
{
  size_t func(void* buf,size_t size, size_t nmemb, void* userp)
  {
    ReadData* rdata = (ReadData*)userp;
    size_t sz = size * nmemb;
    char* cbuf = (char*)buf;
    rdata->d_.insert(rdata->d_.end(),cbuf,cbuf+sz);
    return sz;
  }
}
