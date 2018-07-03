#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

std::ostream& operator<<(std::ostream& s, const CaloRecHit& hit) {
  s << hit.detid().rawId() << ", " << hit.energy() << " GeV, " << hit.time() << " ns ";
  s << " flags=0x" << std::hex << hit.flags() << std::dec << " ";
  s << " aux=0x" << std::hex << hit.aux() << std::dec << " ";
  return s;
}
