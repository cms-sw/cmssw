#ifndef HcalRecAlgos_HcalTDCReco_h
#define HcalRecAlgos_HcalTDCReco_h

#include "DataFormats/HcalRecHit/interface/HBHERecHitFwd.h"
class HcalUpgradeDataFrame;

class HcalTDCReco {
public:
  HcalTDCReco();
  void reconstruct(const HcalUpgradeDataFrame& digi, HBHERecHit& recHit) const;
};
#endif
