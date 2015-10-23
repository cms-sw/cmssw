#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"


HBHERecHit::HBHERecHit() : CaloRecHit(), rawEnergy_(-1.0e21), auxEnergy_(-1.0e21) {
}

HBHERecHit::HBHERecHit(const HcalDetId& id, float energy, float timeRising, float timeFalling) :
  CaloRecHit(id,energy,timeRising),
  timeFalling_(timeFalling),
  rawEnergy_(-1.0e21),
  auxEnergy_(-1.0e21)
{
}

std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit) {
  s << hit.id() << ": " << hit.energy() << " GeV";
  if (hit.eraw() > -0.9e21) {
    s << ", eraw=" << hit.eraw() << " GeV";
  }
  if (hit.eaux() > -0.9e21) {
    s << ", eaux=" << hit.eaux() << " GeV";
  }
  if(hit.time() > -998) {
    s << ", t= " << hit.time() << " to " << hit.timeFalling() << " ns";
  }
  return s;
}

