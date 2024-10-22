#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

//--- The overall probability.  flags is the 32-bit-packed set of flags that
//--- our own concrete implementation of clusterProbability() uses to direct
//--- the computation based on the information stored in the quality word
//--- (and which was computed by the CPE).  The default of flags==0 returns
//--- probX*probY*(1-log(probX*probY)) because of Morris' note.
//--- Flags are static and kept in the transient rec hit.
float SiPixelRecHit::clusterProbability(unsigned int flags) const {
  if (!hasFilledProb()) {
    return 1;
  } else if (flags == 1) {
    return probabilityXY() * probabilityQ();
  } else if (flags == 2) {
    return probabilityQ();
  } else {
    return probabilityXY();
  }
}
