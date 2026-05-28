#ifndef CondFormats_HcalObjects_HcalPulseDelays_h
#define CondFormats_HcalObjects_HcalPulseDelays_h

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalPulseDelay.h"

class HcalPulseDelays : public HcalCondObjectContainer<HcalPulseDelay> {
public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalPulseDelays() : HcalCondObjectContainer<HcalPulseDelay>(nullptr) {}
#endif
  HcalPulseDelays(const HcalTopology* topo) : HcalCondObjectContainer<HcalPulseDelay>(topo) {}

  inline std::string myname() const override { return "HcalPulseDelays"; }

private:
  COND_SERIALIZABLE;
};

#endif  // CondFormats_HcalObjects_HcalPulseDelays_h
