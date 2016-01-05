#ifndef __L1TMUON_PTREFINEMENTUNIT_H__
#define __L1TMUON_PTREFINEMENTUNIT_H__
// 
// Class: L1TMuon::PtRefinementUnit
//
// Info: This is a base class for any algorithm that takes a found track
//       with assigned pT and applies a refinement to that estimate.
//
// Author: L. Gray (FNAL)
//
#include <vector>
#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrackFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace L1TMuon {
  
  class PtRefinementUnit {
  public:
    PtRefinementUnit(const edm::ParameterSet&);
    virtual ~PtRefinementUnit() {}

    virtual void refinePt(const edm::EventSetup&, 
			  InternalTrack&) const = 0;
  protected:
    std::string _name;
  };
}

#endif
