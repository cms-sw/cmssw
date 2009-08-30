#ifndef HBHETIMINGSHAPEDFLAG_GUARD_H
#define HBHETIMINGSHAPEDFLAG_GUARD_H

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"


class HBHETimingShapedFlagSetter {
 public:
  HBHETimingShapedFlagSetter();
  HBHETimingShapedFlagSetter(std::vector<double> tfilterEnvelope);
  HBHETimingShapedFlagSetter(std::vector<double> tfilterEnvelope,
			     bool ignorelowest,
			     bool ignorehighest,
			     double win_offset,
			     double win_gain);
  ~HBHETimingShapedFlagSetter();
  void Clear();
  void SetTimingShapedFlags(HBHERecHit& hbhe);
 private:
  std::vector<double> tfilterEnvelope_;
  bool ignorelowest_;
  bool ignorehighest_;
  double win_offset_;
  double win_gain_;
};

#endif
