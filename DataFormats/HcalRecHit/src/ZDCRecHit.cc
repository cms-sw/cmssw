#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"


ZDCRecHit::ZDCRecHit() : CaloRecHit() {
}

ZDCRecHit::ZDCRecHit(const HcalZDCDetId& id, float energy, float time) :
  CaloRecHit(id,energy,time) {
}

std::ostream& operator<<(std::ostream& s, const ZDCRecHit& hit) {
  return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
}

