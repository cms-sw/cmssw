#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"

ZDCRecHit::ZDCRecHit() : CaloRecHit(), lowGainEnergy_() {}

ZDCRecHit::ZDCRecHit(const HcalZDCDetId& id, float energy, float time, float lowGainEnergy)
    : CaloRecHit(id, energy, time), lowGainEnergy_(lowGainEnergy) {}

std::ostream& operator<<(std::ostream& s, const ZDCRecHit& hit) {
  return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
}
