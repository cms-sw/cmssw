#ifndef RecoTracker_LSTCore_interface_LSTInputHostCollection_h
#define RecoTracker_LSTCore_interface_LSTInputHostCollection_h

#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"

namespace lst {
  // This needs to be PortableHostCollection3 instead of PortableHostMultiCollection for it to work
  using LSTInputHostCollection = PortableHostCollection3<InputHitsSoA, InputPixelHitsSoA, InputPixelSeedsSoA>;
}  // namespace lst

#endif
