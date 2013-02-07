#ifndef DATAFORMATS_HCALRECHIT_HcalDualTimeRECHIT_H
#define DATAFORMATS_HCALRECHIT_HcalDualTimeRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

/** \class HcalDualTimeRecHit
 *  
 * $Date: 2012/11/19 21:26:59 $
 * $Revision: 1.2 $
 *\author R. Wilkinson
 */
class HcalDualTimeRecHit  : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  HcalDualTimeRecHit();
  HcalDualTimeRecHit(const HcalDetId& id, float amplitude, float timeRising, float timeFalling=0);
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

std::ostream& operator<<(std::ostream& s, const HcalDualTimeRecHit& hit);

#endif
