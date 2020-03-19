#ifndef HAVE_XDAQ
#include "CaloOnlineTools/HcalOnlineDb/interface/xdaq_compat.h"

std::string toolbox::toString(const char* format, ...) {
  va_list varlist;
  va_start(varlist, format);
  char tmp[512];
  vsnprintf(tmp, 512, format, varlist);
  return tmp;
}

#endif
