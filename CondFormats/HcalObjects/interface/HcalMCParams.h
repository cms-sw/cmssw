#ifndef HcalMCParams_h
#define HcalMCParams_h


#include "CondFormats/HcalObjects/interface/HcalMCParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalMCParams: public HcalCondObjectContainer<HcalMCParam>
{
 public:
  HcalMCParams():HcalCondObjectContainer<HcalMCParam>() {}

  std::string myname() const {return (std::string)"HcalMCParams";}

 private:

};
#endif
