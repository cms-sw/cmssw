#ifndef HcalPFCorrs_h
#define HcalPFCorrs_h

/*
\class HcalPFCorrs
\author Radek Ofierzynski
POOL object to store PF Corrections
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalPFCorr.h"

//typedef HcalCondObjectContainer<HcalPFCorr> HcalPFCorrs;

class HcalPFCorrs: public HcalCondObjectContainer<HcalPFCorr>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalPFCorrs():HcalCondObjectContainer<HcalPFCorr>(0) {}
#endif
  HcalPFCorrs(const HcalTopology* topo):HcalCondObjectContainer<HcalPFCorr>(topo) {}

  std::string myname() const {return (std::string)"HcalPFCorrs";}

 private:
};

#endif
