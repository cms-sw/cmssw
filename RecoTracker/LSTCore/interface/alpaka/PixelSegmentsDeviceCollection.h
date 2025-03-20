#ifndef RecoTracker_LSTCore_interface_alpaka_PixelSegmentsDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_PixelSegmentsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using PixelSegmentsDeviceCollection = PortableCollection<PixelSegmentsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
