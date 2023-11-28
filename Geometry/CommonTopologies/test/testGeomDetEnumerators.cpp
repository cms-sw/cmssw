#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"

#include <iostream>

int main(void) {
  // Test that each SubDetector is either barrel or endcap

  bool success = true;
  for (int subdetRaw = GeomDetEnumerators::PixelBarrel; subdetRaw != GeomDetEnumerators::invalidDet; ++subdetRaw) {
    auto subdet = static_cast<GeomDetEnumerators::SubDetector>(subdetRaw);
    if (!(GeomDetEnumerators::isBarrel(subdet) || GeomDetEnumerators::isEndcap(subdet))) {
      success = false;
      edm::LogVerbatim("CommonTopologies")
          << "GeomDetEnumerator::SubDetector " << subdet << " (" << subdetRaw << ") is not barrel or endcap!";
    }
    if (GeomDetEnumerators::isBarrel(subdet) && GeomDetEnumerators::isEndcap(subdet)) {
      success = false;
      edm::LogVerbatim("CommonTopologies")
          << "GeomDetEnumerator::SubDetector " << subdet << " (" << subdetRaw << ") is both barrel and endcap!";
    }
  }

  return success ? 0 : 1;
}
