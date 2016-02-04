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
  HcalValidationCorrs():HcalCondObjectContainer<HcalValidationCorr>() {}

  std::string myname() const {return (std::string)"HcalValidationCorrs";}

 private:
};

#endif
