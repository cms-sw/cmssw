#ifndef RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterDevice_H
#define RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterDevice_H

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackMergerCounterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
  template <typename TDev>
  using TrackMergerCounterDevice = PortableDeviceCollection<TDev, TrackMergerCounterSoA>;
}  // namespace reco

#endif  // RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterDevice_H
