#include "DataFormats/HcalRecHit/interface/HORecHit.h"

namespace cms {

  HORecHit::HORecHit() : CaloRecHit() {
  }

  HORecHit::HORecHit(const HcalDetId& id, float energy, float time) :
    CaloRecHit(energy,time),
    id_(id) {
  }
  
  DetId HORecHit::genericId() const { return id_; }

  std::ostream& operator<<(std::ostream& s, const HORecHit& hit) {
    return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  }
}
