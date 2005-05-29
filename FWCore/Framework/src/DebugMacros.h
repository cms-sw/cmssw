#ifndef EDM_DEBUGMACROS_HH
#define EDM_DEBUGMACROS_HH

namespace edm
{
  struct debugvalue
  {
    debugvalue();

    int operator()() { return value_; }
    
    const char* cvalue_;
    int value_;
  };

extern debugvalue debugit;
}

#define FDEBUG(lev) if(lev <= debugit()) std::cerr

#endif

