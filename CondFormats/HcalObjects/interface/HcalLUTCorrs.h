#ifndef HcalLUTCorrs_h
#define HcalLUTCorrs_h

/*
\class HcalLUTCorrs
\author Radek Ofierzynski
POOL object to store LUT Corrections
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalLUTCorr.h"

//typedef HcalCondObjectContainer<HcalLUTCorr> HcalLUTCorrs;

class HcalLUTCorrs: public HcalCondObjectContainer<HcalLUTCorr>
{
 public:
  HcalLUTCorrs():HcalCondObjectContainer<HcalLUTCorr>() {}

  std::string myname() const {return (std::string)"HcalLUTCorrs";}

 private:
};

#endif
