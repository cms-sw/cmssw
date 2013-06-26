#ifndef HcalGainWidths_h
#define HcalGainWidths_h

/** 
\class HcalGainWidths
\author Radek Ofierzynski
POOL container to store GainWidth values 4xCapId
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"

//typedef HcalCondObjectContainer<HcalGainWidth> HcalGainWidths;

class HcalGainWidths: public HcalCondObjectContainer<HcalGainWidth>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalGainWidths():HcalCondObjectContainer<HcalGainWidth>(0) {}
#endif
  HcalGainWidths(const HcalTopology* topo):HcalCondObjectContainer<HcalGainWidth>(topo) {}

  std::string myname() const {return (std::string)"HcalGainWidths";}

 private:
};

#endif
