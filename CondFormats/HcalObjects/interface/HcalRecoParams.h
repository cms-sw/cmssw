#ifndef CondFormatsHcalObjectsHcalRecoParams_h
#define CondFormatsHcalObjectsHcalRecoParams_h


#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalRecoParams: public HcalCondObjectContainer<HcalRecoParam>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalRecoParams():HcalCondObjectContainer<HcalRecoParam>(nullptr) {}
#endif
  HcalRecoParams(const HcalTopology* topo):HcalCondObjectContainer<HcalRecoParam>(topo) {}

  std::string myname() const override {return (std::string)"HcalRecoParams";}

 private:


 COND_SERIALIZABLE;
};
#endif
