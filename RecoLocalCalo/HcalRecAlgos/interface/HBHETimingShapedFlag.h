#ifndef HBHETIMINGSHAPEDFLAG_GUARD_H
#define HBHETIMINGSHAPEDFLAG_GUARD_H

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"


class HBHETimingShapedFlagSetter {
 public:
  HBHETimingShapedFlagSetter();
  HBHETimingShapedFlagSetter(std::vector<double> tfilterEnvelope);
  ~HBHETimingShapedFlagSetter();
  void Clear();
  void SetTimingShapedFlags(HBHERecHit& hbhe);
 private:
  std::vector<double> tfilterEnvelope_;

};

#endif
