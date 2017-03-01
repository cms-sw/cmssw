#ifndef __L1TMUON_CSCCOLLECTOR_H__
#define __L1TMUON_CSCCOLLECTOR_H__
// 
// Class: L1TMuon::CSCCollector
//
// Info: Processes CSC digis into ITMu trigger primitives. 
//       Positional information is not assigned here.
//
// Author: L. Gray (FNAL)
//
#include <vector>
#include "SubsystemCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace L1TMuon {
  
  class CSCCollector: public SubsystemCollector {
  public:
    CSCCollector(const edm::ParameterSet&);
    ~CSCCollector() {}

    virtual void extractPrimitives(const edm::Event&, const edm::EventSetup&, 
				   std::vector<TriggerPrimitive>&) const;
  private:    
  };
}

#endif
