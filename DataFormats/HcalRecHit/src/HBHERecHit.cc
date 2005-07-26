#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"

namespace cms {

  HBHERecHit::HBHERecHit() : CaloRecHit() {
  }

  HBHERecHit::HBHERecHit(const HcalDetId& id, float energy, float time) :
    CaloRecHit(energy,time),
    id_(id) {
  }
  
  DetId HBHERecHit::genericId() const { return id_; }

  std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit) {
    return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  }
}
