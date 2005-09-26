#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

namespace cms {

  CaloRecHit::CaloRecHit() : energy_(0), time_(0) {
  }
  
  CaloRecHit::CaloRecHit(const DetId& id, float energy, float time) : id_(id),energy_(energy), time_(time) {
  }
  
  CaloRecHit::~CaloRecHit() {
  }

  std::ostream& operator<<(std::ostream& s, const CaloRecHit& hit) {
    return s << hit.detid().rawId() << ", " << hit.energy() << " GeV, " << hit.time() << " ns";
  }

}
