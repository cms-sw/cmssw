#ifndef CondFormatsHcalObjectsHcalTPChannelParameters_h
#define CondFormatsHcalObjectsHcalTPChannelParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HcalObjects/interface/HcalTPChannelParameter.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

class HcalTPChannelParameters: public HcalCondObjectContainer<HcalTPChannelParameter> {

public:
  //constructor definition: has to contain 
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalTPChannelParameters():HcalCondObjectContainer<HcalTPChannelParameter>(nullptr) {}
#endif
  HcalTPChannelParameters(const HcalTopology* topo):HcalCondObjectContainer<HcalTPChannelParameter>(topo) {}

  std::string myname() const override {return (std::string)"HcalTPChannelParameters";}

  COND_SERIALIZABLE;
};

#endif
