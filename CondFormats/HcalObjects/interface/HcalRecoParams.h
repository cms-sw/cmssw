#ifndef HcalRecoParams_h
#define HcalRecoParams_h


#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalRecoParams: public HcalCondObjectContainer<HcalRecoParam>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalRecoParams():HcalCondObjectContainer<HcalRecoParam>(0) {}
#endif
  HcalRecoParams(const HcalTopology* topo):HcalCondObjectContainer<HcalRecoParam>(topo) {}

  std::string myname() const {return (std::string)"HcalRecoParams";}

 private:

};
#endif
