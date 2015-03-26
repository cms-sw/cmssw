#ifndef HcalZDCLowGainFractions_h
#define HcalZDCLowGainFractions_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalZDCLowGainFraction.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalZDCLowGainFractions: public HcalCondObjectContainer<HcalZDCLowGainFraction>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalZDCLowGainFractions():HcalCondObjectContainer<HcalZDCLowGainFraction>(0) {}
#endif
  HcalZDCLowGainFractions(const HcalTopology* topo):HcalCondObjectContainer<HcalZDCLowGainFraction>(topo) {}

  std::string myname() const {return (std::string)"HcalZDCLowGainFractions";}

 private:


 COND_SERIALIZABLE;
};
#endif

