#include "DataFormats/HcalRecHit/interface/HFRecHit.h"


HFRecHit::HFRecHit() : CaloRecHit() {
}

HFRecHit::HFRecHit(const HcalDetId& id, float energy, float timeRising, float timeFalling) :
  CaloRecHit(id,energy,timeRising),
  timeFalling_(timeFalling)
{
}

std::ostream& operator<<(std::ostream& s, const HFRecHit& hit) {
  s << hit.id() << ": " << hit.energy() << " GeV";
  if(hit.time() > -998) {
    s << ", t= " << hit.time() << " to " << hit.timeFalling() << " ns";
  }
  return s;
}

