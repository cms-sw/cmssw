#ifndef CondFormats_HcalObjects_HcalPFCuts_h
#define CondFormats_HcalObjects_HcalPFCuts_h

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalPFCut.h"

class HcalPFCuts : public HcalCondObjectContainer<HcalPFCut> {
public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalPFCuts() : HcalCondObjectContainer<HcalPFCut>(nullptr) {}
#endif
  HcalPFCuts(const HcalTopology* topo) : HcalCondObjectContainer<HcalPFCut>(topo) {}

  inline std::string myname() const override { return "HcalPFCuts"; }

private:
  COND_SERIALIZABLE;
};

#endif  // CondFormats_HcalObjects_HcalPFCuts_h
