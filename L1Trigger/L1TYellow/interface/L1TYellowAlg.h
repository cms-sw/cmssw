#ifndef L1TYellowAlg_h
#define L1TYellowAlg_h

#include "CondFormats/L1TYellow/interface/L1TYellowParams.h"
#include "DataFormats/L1TYellow/interface/L1TYellowDigi.h"
#include "DataFormats/L1TYellow/interface/L1TYellowOutput.h"
#include "FWCore/Framework/interface/Event.h"

namespace l1t {

  class L1TYellowAlg { 
  public:
    virtual void processEvent(const L1TYellowDigiCollection &, L1TYellowOutputCollection & out) = 0;    
    virtual ~L1TYellowAlg(){};
  }; 
  
  L1TYellowAlg * NewL1TYellowAlg(const L1TYellowParams & dbPars);
  
} // namespace

#endif
