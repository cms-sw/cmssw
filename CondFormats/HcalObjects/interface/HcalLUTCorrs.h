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
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalLUTCorrs():HcalCondObjectContainer<HcalLUTCorr>(0) {}
#endif
  HcalLUTCorrs(const HcalTopology* topo):HcalCondObjectContainer<HcalLUTCorr>(topo) {}

  std::string myname() const {return (std::string)"HcalLUTCorrs";}

 private:
};

#endif
