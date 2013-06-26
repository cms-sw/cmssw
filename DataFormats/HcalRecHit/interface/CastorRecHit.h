#ifndef DATAFORMATS_HCALRECHIT_CASTORRECHIT_H
#define DATAFORMATS_HCALRECHIT_CASTORRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


class CastorRecHit : public CaloRecHit {
public:
  typedef HcalCastorDetId key_type;

  CastorRecHit();
  CastorRecHit(const HcalCastorDetId& id, float energy, float time);
  /// get the id
  HcalCastorDetId id() const { return HcalCastorDetId(detid()); }
};

std::ostream& operator<<(std::ostream& s, const CastorRecHit& hit);

#endif

