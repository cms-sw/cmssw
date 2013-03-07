#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"


HBHERecHit::HBHERecHit() : CaloRecHit() {
}

// HBHERecHit::HBHERecHit(const HcalDetId& id, float energy, float time) :
//   CaloRecHit(id,energy,time) {
// }

//std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit) {
//  return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
//}

HBHERecHit::HBHERecHit(const HcalDetId& id, float energy, float timeRising, float timeFalling) :
  CaloRecHit(id,energy,timeRising),
  timeFalling_(timeFalling)	
{
}

std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit) {
  s << hit.id() << ": " << hit.energy() << " GeV";
  if(hit.time() > -998) {
    s << ", t= " << hit.time() << " to " << hit.timeFalling() << " ns";
  }
  return s;
}

