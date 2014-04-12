#ifndef HcalGains_h
#define HcalGains_h

/** 
\class HcalGains
\author Radek Ofierzynski
POOL container to store Gain values 4xCapId
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"

//typedef HcalCondObjectContainer<HcalGain> HcalGains;

class HcalGains: public HcalCondObjectContainer<HcalGain>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalGains():HcalCondObjectContainer<HcalGain>(0) {}
#endif
  HcalGains(const HcalTopology* topo):HcalCondObjectContainer<HcalGain>(topo) {}

  std::string myname() const {return (std::string)"HcalGains";}

 private:
};

#endif
