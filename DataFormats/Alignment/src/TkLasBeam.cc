#include "DataFormats/Alignment/interface/TkLasBeam.h"
#include "FWCore/Utilities/interface/Exception.h"

bool TkLasBeam::isTecInternal(int side) const {
  switch (side) {
    case 0:
      return beamId % 1000 / 100 < 2;
    case -1:
      return beamId % 1000 / 100 == 1;
    case 1:
      return beamId % 1000 / 100 == 0;
  }

  throw cms::Exception("[TkLasBeam::isTecInternal]") << " ** ERROR: side=" << side << " undefined, must be -1, 0 or 1.";
  return false;  // unreached
}
