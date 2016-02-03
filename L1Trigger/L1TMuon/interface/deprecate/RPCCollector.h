#ifndef __L1TMUON_RPCCOLLECTOR_H__
#define __L1TMUON_RPCCOLLECTOR_H__
// 
// Class: L1TMuon::RPCCollector
//
// Info: Processes RPC digis into L1TMuon trigger primitives. 
//       Positional information is not assigned here.
//
// Author: L. Gray (FNAL)
//
#include <vector>
#include "SubsystemCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace L1TMuon {
  
  class RPCCollector: public SubsystemCollector {
  public:
    RPCCollector(const edm::ParameterSet&);
    ~RPCCollector() {}

    virtual void extractPrimitives(const edm::Event&, const edm::EventSetup&, 
				   std::vector<TriggerPrimitive>&) const;
  private:    
  };
}

#endif
