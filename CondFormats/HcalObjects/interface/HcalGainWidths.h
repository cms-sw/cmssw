#ifndef HcalGainWidths_h
#define HcalGainWidths_h

/** 
\class HcalGainWidths
\author Radek Ofierzynski
POOL container to store GainWidth values 4xCapId
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"

//typedef HcalCondObjectContainer<HcalGainWidth> HcalGainWidths;

class HcalGainWidths: public HcalCondObjectContainer<HcalGainWidth>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalGainWidths():HcalCondObjectContainer<HcalGainWidth>(nullptr) {}
#endif
  HcalGainWidths(const HcalTopology* topo):HcalCondObjectContainer<HcalGainWidth>(topo) {}

  std::string myname() const override {return (std::string)"HcalGainWidths";}

 private:

 COND_SERIALIZABLE;
};

#endif
