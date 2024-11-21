#ifndef RecoTracker_LSTCore_interface_TripletsDeviceCollection_h
#define RecoTracker_LSTCore_interface_TripletsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using TripletsDeviceCollection = PortableCollection2<TripletsSoA, TripletsOccupancySoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
