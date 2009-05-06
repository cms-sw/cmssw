#ifndef HcalTimeCorrs_h
#define HcalTimeCorrs_h

/*
\class HcalTimeCorrs
\author Radek Ofierzynski
POOL object to store time offsets
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorr.h"

//typedef HcalCondObjectContainer<HcalTimeCorr> HcalTimeCorrs;

class HcalTimeCorrs: public HcalCondObjectContainer<HcalTimeCorr>
{
 public:
  HcalTimeCorrs():HcalCondObjectContainer<HcalTimeCorr>() {}

  std::string myname() const {return (std::string)"HcalTimeCorrs";}

 private:
};

#endif
