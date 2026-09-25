#include "DataFormats/HcalRecHit/interface/HFRecHit.h"

namespace io_v1 {
  std::ostream& operator<<(std::ostream& s, const HFRecHit& hit) {
    s << hit.id() << ": " << hit.energy() << " GeV";
    if (hit.time() > -998) {
      s << ", t= " << hit.time() << " to " << hit.timeFalling() << " ns";
    }
    return s;
  }
}  // namespace io_v1
