#ifndef HcalTimingParams_h
#define HcalTimingParams_h


#include "CondFormats/HcalObjects/interface/HcalTimingParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalTimingParams: public HcalCondObjectContainer<HcalTimingParam>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalTimingParams():HcalCondObjectContainer<HcalTimingParam>(0) {}
#endif
  HcalTimingParams(const HcalTopology* topo):HcalCondObjectContainer<HcalTimingParam>(topo) {}

  std::string myname() const {return (std::string)"HcalTimingParams";}

 private:

};
#endif
