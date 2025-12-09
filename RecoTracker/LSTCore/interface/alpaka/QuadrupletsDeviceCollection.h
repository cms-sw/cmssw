#ifndef RecoTracker_LSTCore_interface_QuadrupletsDeviceCollection_h
#define RecoTracker_LSTCore_interface_QuadrupletsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/QuadrupletsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using QuadrupletsDeviceCollection = PortableCollection2<QuadrupletsSoA, QuadrupletsOccupancySoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
