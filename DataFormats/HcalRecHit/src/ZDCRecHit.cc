#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"

ZDCRecHit::ZDCRecHit()
    : CaloRecHit(), lowGainEnergy_(), energySOIp1_(-99), ratioSOIp1_(-99), TDCtime_(-99), chargeWeightedTime_(-99) {}

ZDCRecHit::ZDCRecHit(const HcalZDCDetId& id, float energy, float time, float lowGainEnergy)
    : CaloRecHit(id, energy, time),
      lowGainEnergy_(lowGainEnergy),
      energySOIp1_(-99),
      ratioSOIp1_(-99),
      TDCtime_(-99),
      chargeWeightedTime_(-99) {}

std::ostream& operator<<(std::ostream& s, const ZDCRecHit& hit) {
  return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
}
