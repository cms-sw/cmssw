#ifndef HcalLUTCorrs_h
#define HcalLUTCorrs_h

/*
\class HcalLUTCorrs
\author Radek Ofierzynski
POOL object to store LUT Corrections
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalLUTCorr.h"

//typedef HcalCondObjectContainer<HcalLUTCorr> HcalLUTCorrs;

class HcalLUTCorrs: public HcalCondObjectContainer<HcalLUTCorr>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalLUTCorrs():HcalCondObjectContainer<HcalLUTCorr>(nullptr) {}
#endif
  HcalLUTCorrs(const HcalTopology* topo):HcalCondObjectContainer<HcalLUTCorr>(topo) {}

  std::string myname() const override {return (std::string)"HcalLUTCorrs";}

 private:

 COND_SERIALIZABLE;
};

#endif
