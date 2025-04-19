#ifndef RecoTracker_LSTCore_interface_alpaka_LSTInputDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_LSTInputDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using LSTInputDeviceCollection = PortableCollection2<HitsBaseSoA, PixelSeedsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
