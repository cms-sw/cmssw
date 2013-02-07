#include "DataFormats/HcalRecHit/interface/HcalDualTimeRecHit.h"


HcalDualTimeRecHit::HcalDualTimeRecHit() : CaloRecHit() {
}

HcalDualTimeRecHit::HcalDualTimeRecHit(const HcalDetId& id, float energy, float timeRising, float timeFalling) :
  CaloRecHit(id,energy,timeRising),
  timeFalling_(timeFalling) 
{
}

std::ostream& operator<<(std::ostream& s, const HcalDualTimeRecHit& hit) {
  s << hit.id() << ": " << hit.energy() << " GeV";
  if(hit.time() > -998) {
    s << ", t= " << hit.time() << " to " << hit.timeFalling() << " ns";
  }
  return s;
}

