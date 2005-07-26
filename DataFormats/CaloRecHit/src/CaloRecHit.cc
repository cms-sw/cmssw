#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

namespace cms {

  CaloRecHit::CaloRecHit() : energy_(0), time_(0) {
  }
  
  CaloRecHit::CaloRecHit(float energy, float time) : energy_(energy), time_(time) {
  }
  
  CaloRecHit::~CaloRecHit() {
  }

  std::ostream& operator<<(std::ostream& s, const CaloRecHit& hit) {
    return s << hit.genericId().rawId() << ", " << hit.energy() << " GeV, " << hit.time() << " ns";
  }

}
