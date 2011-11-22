#ifndef DATAFORMATS_HCALRECHIT_HcalUpgradeRECHIT_H
#define DATAFORMATS_HCALRECHIT_HcalUpgradeRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

/** \class HcalUpgradeRecHit
 *  
 * $Date: 2006/08/15 15:49:21 $
 * $Revision: 1.1 $
 *\author R. Wilkinson
 */
class HcalUpgradeRecHit  : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  HcalUpgradeRecHit();
  HcalUpgradeRecHit(const HcalDetId& id, float amplitude, float timeRising, float timeFalling=0);
  /// get the amplitude (generally fC, but can vary)
  /// get the hit time (if available)
  float timeFalling() const { return timeFalling_; }
  /// get the id
  /// get the id
  HcalDetId id() const { return HcalDetId(detid()); }

private:
  HcalDetId id_;
  float timeFalling_;
};

std::ostream& operator<<(std::ostream& s, const HcalUpgradeRecHit& hit);

#endif
