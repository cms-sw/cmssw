#include "DataFormats/HcalRecHit/interface/HcalTriggerPrimitiveRecHit.h"

namespace cms {

  HcalTriggerPrimitiveRecHit::HcalTriggerPrimitiveRecHit() : CaloRecHit() {
  }

  HcalTriggerPrimitiveRecHit::HcalTriggerPrimitiveRecHit(const HcalTrigTowerDetId& id, float energy, float time, int bunch, int index, int n) :
    CaloRecHit(energy,time),
    id_(id),
    bunch_(bunch),
    index_(index),
    count_(n)
  {
  }
  
  DetId HcalTriggerPrimitiveRecHit::genericId() const { return id_; }

  std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveRecHit& hit) {
    return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  }
}
