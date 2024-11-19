#ifndef RecoTracker_LSTCore_interface_SegmentsHostCollection_h
#define RecoTracker_LSTCore_interface_SegmentsHostCollection_h

#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using SegmentsHostCollection = PortableHostMultiCollection<SegmentsSoA, SegmentsOccupancySoA, SegmentsPixelSoA>;
}  // namespace lst
#endif
