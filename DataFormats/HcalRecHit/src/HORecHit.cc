#include "DataFormats/HcalRecHit/interface/HORecHit.h"

namespace io_v1 {
  std::ostream& operator<<(std::ostream& s, const HORecHit& hit) {
    return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  }
}  // namespace io_v1
