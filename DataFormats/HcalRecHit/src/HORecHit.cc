#include "DataFormats/HcalRecHit/interface/HORecHit.h"

namespace cms {

  HORecHit::HORecHit() : CaloRecHit() {
  }

  HORecHit::HORecHit(const HcalDetId& id, float energy, float time) :
    CaloRecHit(id,energy,time){
  }

  std::ostream& operator<<(std::ostream& s, const HORecHit& hit) {
    return s << hit.hcal_id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  }
}
