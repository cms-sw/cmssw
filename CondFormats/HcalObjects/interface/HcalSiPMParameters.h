#ifndef CondFormatsHcalObjectsHcalSiPMParameters_h
#define CondFormatsHcalObjectsHcalSiPMParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMParameter.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

class HcalSiPMParameters: public HcalCondObjectContainer<HcalSiPMParameter> {

public:
  //constructor definition: has to contain 
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalSiPMParameters():HcalCondObjectContainer<HcalSiPMParameter>(nullptr) {}
#endif
  HcalSiPMParameters(const HcalTopology* topo):HcalCondObjectContainer<HcalSiPMParameter>(topo) {}

  std::string myname() const override {return (std::string)"HcalSiPMParameters";}

  COND_SERIALIZABLE;
};

#endif
