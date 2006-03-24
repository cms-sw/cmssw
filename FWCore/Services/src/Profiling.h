#ifndef Services_SimpleProfiling_h
#define Services_SimpleProfiling_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     SimpleProfiling
// 
//
// Original Author:  Jim Kowalkowski
// $Id:$
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  namespace service {
    class SimpleProfiling
    {
    public:
      SimpleProfiling(const ParameterSet&,ActivityRegistry&);
      ~SimpleProfiling();
      
      void postBeginJob();
      void postEndJob();
      
    private:
    };
  }
}



#endif
