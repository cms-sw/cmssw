#ifndef FWCore_Framework_BeginJobCleanup_h
#define FWCore_Framework_BeginJobCleanup_h

#include <set>
#include <string>

namespace edm {

  
  inline
  std::set<std::string>& allModuleNames() {
    static std::set<std::string> allModuleNames_;
    return allModuleNames_;
  }
}

#endif
