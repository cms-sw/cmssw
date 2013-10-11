#ifndef YellowAlg_h
#define YellowAlg_h

#include "CondFormats/L1TYellow/interface/YellowParams.h"
#include "DataFormats/L1TYellow/interface/YellowDigi.h"
#include "DataFormats/L1TYellow/interface/YellowOutput.h"
#include "FWCore/Framework/interface/Event.h"

namespace l1t {

  class YellowAlg { 
  public:
    virtual void processEvent(const YellowDigiCollection &, YellowOutputCollection & out) = 0;    
    virtual ~YellowAlg(){};
  }; 
  
  YellowAlg * NewYellowAlg(const YellowParams & dbPars);
  
} // namespace

#endif
