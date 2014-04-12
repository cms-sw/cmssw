#ifndef HcalRespCorrs_h
#define HcalRespCorrs_h

/*
\class HcalRespCorrs
\author Radek Ofierzynski
POOL object to store Hcal Response Corrections
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorr.h"

//typedef HcalCondObjectContainer<HcalRespCorr> HcalRespCorrs;

class HcalRespCorrs: public HcalCondObjectContainer<HcalRespCorr>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalRespCorrs():HcalCondObjectContainer<HcalRespCorr>(0) {}
#endif
  HcalRespCorrs(const HcalTopology* topo):HcalCondObjectContainer<HcalRespCorr>(topo) {}

  std::string myname() const {return (std::string)"HcalRespCorrs";}

 private:
};

#endif
