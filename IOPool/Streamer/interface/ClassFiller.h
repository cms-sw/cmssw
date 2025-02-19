#ifndef IOPool_Streamer_ClassFiller_h
#define IOPool_Streamer_ClassFiller_h

// -*- C++ -*-

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "Rtypes.h"

#include <typeinfo>
#include <string>
#include <set>

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
  void loadCap(const std::string& name);
  void doBuildRealData(const std::string& name);
}

#endif
