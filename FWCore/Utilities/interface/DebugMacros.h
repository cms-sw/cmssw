#ifndef FWCore_Utilities_DebugMacros_h
#define FWCore_Utilities_DebugMacros_h
#include <iostream>

namespace edm {
  struct debugvalue {

    debugvalue();

    int operator()() { return value_; }
    
    const char* cvalue_;
    int value_;
  };

[[cms::thread_safe]] extern debugvalue debugit;
}

#define FDEBUG(lev) if(lev <= debugit()) std::cerr

#endif
