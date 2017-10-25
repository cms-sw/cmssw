#ifndef HcalMCParams_h
#define HcalMCParams_h


#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalMCParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalMCParams: public HcalCondObjectContainer<HcalMCParam>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalMCParams():HcalCondObjectContainer<HcalMCParam>(nullptr) {}
#endif
  HcalMCParams(const HcalTopology* topo):HcalCondObjectContainer<HcalMCParam>(topo) {}

  std::string myname() const override {return (std::string)"HcalMCParams";}

 private:


 COND_SERIALIZABLE;
};
#endif
