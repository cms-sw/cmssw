#ifndef ZDCLowGainFractions_h
#define ZDCLowGainFractions_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/ZDCLowGainFraction.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class ZDCLowGainFractions: public HcalCondObjectContainer<ZDCLowGainFraction>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  ZDCLowGainFractions():HcalCondObjectContainer<ZDCLowGainFraction>(0) {}
#endif
  ZDCLowGainFractions(const HcalTopology* topo):HcalCondObjectContainer<ZDCLowGainFraction>(topo) {}

  std::string myname() const {return (std::string)"ZDCLowGainFractions";}

 private:


 COND_SERIALIZABLE;
};
#endif

