#ifndef HcalZSThresholds_h
#define HcalZSThresholds_h

/*
\class HcalZSThresholds
\author Radek Ofierzynski
POOL object to store Zero Suppression Thresholds
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalZSThreshold.h"

//typedef HcalCondObjectContainer<HcalZSThreshold> HcalZSThresholds;

class HcalZSThresholds: public HcalCondObjectContainer<HcalZSThreshold>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalZSThresholds():HcalCondObjectContainer<HcalZSThreshold>(0) {}
#endif
  HcalZSThresholds(const HcalTopology* topo):HcalCondObjectContainer<HcalZSThreshold>(topo) {}

  std::string myname() const {return (std::string)"HcalZSThresholds";}

 private:
};

#endif
