#ifndef HcalLongRecoParams_h
#define HcalLongRecoParams_h


#include "CondFormats/HcalObjects/interface/HcalLongRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalLongRecoParams: public HcalCondObjectContainer<HcalLongRecoParam>
{
 public:
  HcalLongRecoParams():HcalCondObjectContainer<HcalLongRecoParam>() {}

  std::string myname() const {return (std::string)"HcalLongRecoParams";}

 private:

};
#endif
