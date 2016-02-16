#ifndef __L1TMUON_DTCOLLECTOR_H__
#define __L1TMUON_DTCOLLECTOR_H__
//
// Class: L1TMuon::DTCollector
//
// Info: Processes the DT digis into L1TMuon trigger primitives.
//       Positional information is not assigned here.
//
// Author: L. Gray (FNAL)
//
#include <vector>
#include <memory>
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/DTBunchCrossingCleaner.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;

namespace L1TwinMux {

  class DTCollector {
  public:
    DTCollector();
    ~DTCollector() {}

    //virtual void extractPrimitives(const edm::Event&, const edm::EventSetup&,
	//			   std::vector<TriggerPrimitive>&) const;
	virtual void extractPrimitives(edm::Handle<L1MuDTChambPhContainer> phiDigis,
                                    edm::Handle<L1MuDTChambThContainer> thetaDigis,
                                    TriggerPrimitiveCollection& out) const;

  private:
    TriggerPrimitive processDigis(const L1MuDTChambPhDigi&,
				  const int &segment_number) const;
    TriggerPrimitive processDigis(const L1MuDTChambThDigi&,
				  const int bti_group) const;
    TriggerPrimitive processDigis(const L1MuDTChambPhDigi&,
				  const L1MuDTChambThDigi&,
				  const int bti_group) const;
    int findBTIGroupForThetaDigi(const L1MuDTChambThDigi&,
				 const int position) const;
    const int bx_min = -9, bx_max = 7;
    std::unique_ptr<DTBunchCrossingCleaner> _bxc;
  };
}

#endif
