#ifndef RecoTracker_LSTCore_interface_PixelSegmentsHostCollection_h
#define RecoTracker_LSTCore_interface_PixelSegmentsHostCollection_h

#include "RecoTracker/LSTCore/interface/PixelSegmentsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using PixelSegmentsHostCollection = PortableHostCollection<PixelSegmentsSoA>;
}  // namespace lst
#endif
