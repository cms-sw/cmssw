#ifndef HcalZSThresholds_h
#define HcalZSThresholds_h

/*
\class HcalZSThresholds
\author Radek Ofierzynski
POOL object to store Zero Suppression Thresholds
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalZSThreshold.h"

//typedef HcalCondObjectContainer<HcalZSThreshold> HcalZSThresholds;

class HcalZSThresholds: public HcalCondObjectContainer<HcalZSThreshold>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalZSThresholds():HcalCondObjectContainer<HcalZSThreshold>(nullptr) {}
#endif
  HcalZSThresholds(const HcalTopology* topo):HcalCondObjectContainer<HcalZSThreshold>(topo) {}

  std::string myname() const override {return (std::string)"HcalZSThresholds";}

 private:

 COND_SERIALIZABLE;
};

#endif
