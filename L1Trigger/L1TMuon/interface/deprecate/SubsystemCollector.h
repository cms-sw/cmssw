#ifndef __L1TMUON_SUBSYSTEMCOLLECTOR_H__
#define __L1TMUON_SUBSYSTEMCOLLECTOR_H__
// 
// Class: L1TMuon::SubsystemCollector
//
// Info: This is the base class for a object that eats a specified subsystem
//       and turns those digis into L1ITMu::TriggerPrimitives
//
// Author: L. Gray (FNAL)
//
#include <vector>
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace L1TMuon {
  
  class SubsystemCollector {
  public:
    SubsystemCollector(const edm::ParameterSet&);
    virtual ~SubsystemCollector() {}

    virtual void extractPrimitives(const edm::Event&, const edm::EventSetup&, 
				   std::vector<TriggerPrimitive>&) const = 0;
  protected:
    edm::InputTag _src;
  };
}

#endif
