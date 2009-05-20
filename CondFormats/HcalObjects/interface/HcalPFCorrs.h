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
  HcalPFCorrs():HcalCondObjectContainer<HcalPFCorr>() {}

  std::string myname() const {return (std::string)"HcalPFCorrs";}

 private:
};

#endif
