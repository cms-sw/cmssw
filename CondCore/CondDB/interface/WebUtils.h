#ifndef CondCore_CondDB_WebUtils_h
#define CondCore_CondDB_WebUtils_h

#include <string>

namespace cond {

  unsigned long httpGet(const std::string& urlString, std::string& info);

}

#endif
