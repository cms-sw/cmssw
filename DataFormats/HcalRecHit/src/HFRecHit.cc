#include "DataFormats/HcalRecHit/interface/HFRecHit.h"

namespace cms {

  HFRecHit::HFRecHit() : CaloRecHit() {
  }

  HFRecHit::HFRecHit(const HcalDetId& id, float energy, float time) :
    CaloRecHit(id,energy,time) {
  }

  std::ostream& operator<<(std::ostream& s, const HFRecHit& hit) {
    return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  }
}
