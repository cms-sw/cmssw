#include "DataFormats/HcalRecHit/interface/HFRecHit.h"


HFRecHit::HFRecHit() : 
  CaloRecHit(),
  timeFalling_(0.f),
  auxHF_(0)
{
}

HFRecHit::HFRecHit(const HcalDetId& id, float energy, float timeRising, float timeFalling) :
  CaloRecHit(id,energy,timeRising),
  timeFalling_(timeFalling),
  auxHF_(0)
{
}

std::ostream& operator<<(std::ostream& s, const HFRecHit& hit) {
  s << hit.id() << ": " << hit.energy() << " GeV";
  if(hit.time() > -998) {
    s << ", t= " << hit.time() << " to " << hit.timeFalling() << " ns";
  }
  return s;
}

