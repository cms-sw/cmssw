#ifndef HcalValidationCorrs_h
#define HcalValidationCorrs_h

/*
\class HcalValidationCorrs
\author Gena Kukartsev kukarzev@fnal.gov
POOL object to store HCAL validation corrections
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalValidationCorr.h"

//typedef HcalCondObjectContainer<HcalValidationCorr> HcalValidationCorrs;

class HcalValidationCorrs: public HcalCondObjectContainer<HcalValidationCorr>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalValidationCorrs():HcalCondObjectContainer<HcalValidationCorr>(0) {}
#endif
  HcalValidationCorrs(const HcalTopology* topo):HcalCondObjectContainer<HcalValidationCorr>(topo) {}

  std::string myname() const {return (std::string)"HcalValidationCorrs";}

 private:
};

#endif
