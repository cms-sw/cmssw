#ifndef HcalLongRecoParams_h
#define HcalLongRecoParams_h


#include "CondFormats/HcalObjects/interface/HcalLongRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalLongRecoParams: public HcalCondObjectContainer<HcalLongRecoParam>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalLongRecoParams():HcalCondObjectContainer<HcalLongRecoParam>(0) {}
#endif
  HcalLongRecoParams(const HcalTopology* topo):HcalCondObjectContainer<HcalLongRecoParam>(topo) {}

  std::string myname() const {return (std::string)"HcalLongRecoParams";}

 private:

};
#endif
