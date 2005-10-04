#include "DataFormats/HcalRecHit/interface/HcalTriggerPrimitiveRecHit.h"


HcalTriggerPrimitiveRecHit::HcalTriggerPrimitiveRecHit() : CaloRecHit() {
}

HcalTriggerPrimitiveRecHit::HcalTriggerPrimitiveRecHit(const HcalTrigTowerDetId& id, float energy, float time, int bunch, int index, int n) :
  CaloRecHit(id,energy,time),
  bunch_(bunch),
  index_(index),
  count_(n)
{
}

std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveRecHit& hit) {
  return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
}

