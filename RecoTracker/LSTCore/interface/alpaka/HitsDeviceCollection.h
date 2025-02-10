#ifndef RecoTracker_LSTCore_interface_alpaka_HitsDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_HitsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/HitsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using HitsDeviceCollection = PortableCollection2<HitsSoA, HitsRangesSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
