#ifndef HcalMCParams_h
#define HcalMCParams_h


#include "CondFormats/HcalObjects/interface/HcalMCParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalMCParams: public HcalCondObjectContainer<HcalMCParam>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalMCParams():HcalCondObjectContainer<HcalMCParam>(0) {}
#endif
  HcalMCParams(const HcalTopology* topo):HcalCondObjectContainer<HcalMCParam>(topo) {}

  std::string myname() const {return (std::string)"HcalMCParams";}

 private:

};
#endif
