#ifndef L1TYellowAlg_h
#define L1TYellowAlg_h

#include "L1Trigger/L1TYellow/interface/L1TYellowDBParams.h"
#include "DataFormats/L1TYellow/interface/L1TYellowOutput.h"
#include "FWCore/Framework/interface/Event.h"

namespace l1t {

  class L1TYellowAlg { 
  public:
    virtual void processEvent(edm::Event& iEvent, L1TYellowOutputCollection & out) = 0;    
    virtual ~L1TYellowAlg(){};
  }; 
  
  L1TYellowAlg * NewL1TYellowAlg(const L1TYellowDBParams & dbPars);
  
} // namespace

#endif
