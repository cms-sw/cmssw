#include "DataFormats/HcalRecHit/interface/HORecHit.h"

std::ostream& operator<<(std::ostream& s, const HORecHit& hit) {
  return s << hit.id() << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
}
