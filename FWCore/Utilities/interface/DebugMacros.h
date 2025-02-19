#ifndef Utilities_DebugMacros_h
#define Utilities_DebugMacros_h

namespace edm {
  struct debugvalue {

    debugvalue();

    int operator()() { return value_; }
    
    const char* cvalue_;
    int value_;
  };

extern debugvalue debugit;
}

#define FDEBUG(lev) if(lev <= debugit()) std::cerr

#endif
