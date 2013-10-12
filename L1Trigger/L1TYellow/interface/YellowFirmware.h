#ifndef YELLOWFIRMWARE_H
#define YELLOWFIRMWARE_H

#include "CondFormats/L1TYellow/interface/YellowParams.h"
#include "DataFormats/L1TYellow/interface/YellowDigi.h"
#include "DataFormats/L1TYellow/interface/YellowOutput.h"
#include "FWCore/Framework/interface/Event.h"

namespace l1t {
    
  class YellowFirmware { 
  public:
    virtual void processEvent(const YellowDigiCollection &, YellowOutputCollection & out) = 0;    
    virtual ~YellowFirmware(){};
  }; 
  
} // namespace

#endif
