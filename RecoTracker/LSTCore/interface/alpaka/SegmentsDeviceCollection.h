#ifndef RecoTracker_LSTCore_interface_alpaka_SegmentsDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_SegmentsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using SegmentsDeviceCollection = PortableCollection3<SegmentsSoA, SegmentsOccupancySoA, SegmentsPixelSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
