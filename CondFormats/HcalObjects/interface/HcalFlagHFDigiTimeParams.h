#ifndef GUARD_HcalFlagHFDigiTimeParams_h
#define GUARD_HcalFlagHFDigiTimeParams_h

#include "CondFormats/HcalObjects/interface/HcalFlagHFDigiTimeParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalFlagHFDigiTimeParams: public HcalCondObjectContainer<HcalFlagHFDigiTimeParam>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalFlagHFDigiTimeParams():HcalCondObjectContainer<HcalFlagHFDigiTimeParam>(0) {}
#endif
  HcalFlagHFDigiTimeParams(const HcalTopology* ht):HcalCondObjectContainer<HcalFlagHFDigiTimeParam>(ht) {}

  std::string myname() const {return (std::string)"HcalFlagHFDigiTimeParams";}

 private:

};
#endif
