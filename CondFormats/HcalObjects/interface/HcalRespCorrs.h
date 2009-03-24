#ifndef HcalRespCorrs_h
#define HcalRespCorrs_h

/*
\class HcalRespCorrs
\author Radek Ofierzynski
POOL object to store Zero Suppression Thresholds
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorr.h"

//typedef HcalCondObjectContainer<HcalRespCorr> HcalRespCorrs;

class HcalRespCorrs: public HcalCondObjectContainer<HcalRespCorr>
{
 public:
  HcalRespCorrs():HcalCondObjectContainer<HcalRespCorr>() {}

  std::string myname() const {return (std::string)"HcalRespCorrs";}

 private:
};

#endif
