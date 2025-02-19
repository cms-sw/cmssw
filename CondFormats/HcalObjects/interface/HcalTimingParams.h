#ifndef HcalTimingParams_h
#define HcalTimingParams_h


#include "CondFormats/HcalObjects/interface/HcalTimingParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalTimingParams: public HcalCondObjectContainer<HcalTimingParam>
{
 public:
  HcalTimingParams():HcalCondObjectContainer<HcalTimingParam>() {}

  std::string myname() const {return (std::string)"HcalTimingParams";}

 private:

};
#endif
