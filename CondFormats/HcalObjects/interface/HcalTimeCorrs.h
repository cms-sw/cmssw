#ifndef HcalTimeCorrs_h
#define HcalTimeCorrs_h

/*
\class HcalTimeCorrs
\author Radek Ofierzynski
POOL object to store time offsets
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorr.h"

//typedef HcalCondObjectContainer<HcalTimeCorr> HcalTimeCorrs;

class HcalTimeCorrs: public HcalCondObjectContainer<HcalTimeCorr>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalTimeCorrs():HcalCondObjectContainer<HcalTimeCorr>(nullptr) {}
#endif
  HcalTimeCorrs(const HcalTopology* topo):HcalCondObjectContainer<HcalTimeCorr>(topo) {}

  std::string myname() const override {return (std::string)"HcalTimeCorrs";}

 private:

 COND_SERIALIZABLE;
};

#endif
