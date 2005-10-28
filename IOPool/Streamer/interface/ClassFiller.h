#ifndef Streamer_ClassFiller_h
#define Streamer_ClassFiller_h

#include "FWCore/Utilities/interface/DebugMacros.h"

#include "TClass.h"
#include "TBuffer.h"

#include <typeinfo>

namespace edm
{
  class RootDebug
  {
  public:
    RootDebug(int flevel, int rlevel):
      flevel_(flevel),rlevel_(rlevel),old_(gDebug)
    { if(flevel_ < debugit()) gDebug=rlevel_; }
    ~RootDebug()
    { if(flevel_ < debugit()) gDebug=old_; } 
    
  private:
    int flevel_;
    int rlevel_;
    int old_;
  };

  void loadExtraClasses();
  TClass* getTClass(const std::type_info& ti);
}

#endif
