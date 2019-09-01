#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

CastorRecHit::CastorRecHit() : CaloRecHit() {}

CastorRecHit::CastorRecHit(const HcalCastorDetId& id, float energy, float time) : CaloRecHit(id, energy, time) {}

std::ostream& operator<<(std::ostream& s, const CastorRecHit& hit) {
  return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
}
