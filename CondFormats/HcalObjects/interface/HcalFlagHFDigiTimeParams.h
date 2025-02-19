#ifndef GUARD_HcalFlagHFDigiTimeParams_h
#define GUARD_HcalFlagHFDigiTimeParams_h

#include "CondFormats/HcalObjects/interface/HcalFlagHFDigiTimeParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalFlagHFDigiTimeParams: public HcalCondObjectContainer<HcalFlagHFDigiTimeParam>
{
 public:
  HcalFlagHFDigiTimeParams():HcalCondObjectContainer<HcalFlagHFDigiTimeParam>() {}

  std::string myname() const {return (std::string)"HcalFlagHFDigiTimeParams";}

 private:

};
#endif
