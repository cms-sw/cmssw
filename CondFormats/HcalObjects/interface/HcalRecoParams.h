#ifndef HcalRecoParams_h
#define HcalRecoParams_h


#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalRecoParams: public HcalCondObjectContainer<HcalRecoParam>
{
 public:
  HcalRecoParams():HcalCondObjectContainer<HcalRecoParam>() {}

  std::string myname() const {return (std::string)"HcalRecoParams";}

 private:

};
#endif
