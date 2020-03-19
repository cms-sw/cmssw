#ifndef Utilities_DebugMacros_h
#define Utilities_DebugMacros_h

#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace edm {
  struct debugvalue {
    debugvalue();

    int operator()() { return value_; }

    const char* cvalue_;
    int value_;
  };

  CMS_THREAD_SAFE extern debugvalue debugit;
}  // namespace edm

#define FDEBUG(lev)     \
  if (lev <= debugit()) \
  std::cerr

#endif
