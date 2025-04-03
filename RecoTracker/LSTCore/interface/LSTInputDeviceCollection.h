#ifndef RecoTracker_LSTCore_interface_LSTInputDeviceCollection_h
#define RecoTracker_LSTCore_interface_LSTInputDeviceCollection_h

#include <alpaka/alpaka.hpp>

#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"

namespace lst {
  template <typename TDev>
  using LSTInputDeviceCollection =
      PortableDeviceMultiCollection<TDev, InputHitsSoA, InputPixelHitsSoA, InputPixelSeedsSoA>;
}  // namespace lst

#endif
