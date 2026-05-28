#include "DataFormats/HGCalReco/interface/throwOutOfBoundsException.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace ticllayer {
  void throwOutOfBoundsException(float eta, float phi, int bin, int size) noexcept(false) {
    cms::Exception e("TiclOutOfBound");
    e.format("bin index {} is out of bounds (container size is {}). It was generated from eta: {} phi: {}",
             bin,
             size,
             eta,
             phi);
    throw e;
  }
}  // namespace ticllayer
