#ifndef FWCore_Services_Profiling_h
#define FWCore_Services_Profiling_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     SimpleProfiling
// 
//
// Original Author:  Jim Kowalkowski
// $Id: Profiling.h,v 1.1 2006/03/11 04:28:21 jbk Exp $
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
