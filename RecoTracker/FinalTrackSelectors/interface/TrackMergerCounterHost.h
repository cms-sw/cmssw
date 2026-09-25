#ifndef RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterHost_h
#define RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackMergerCounterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
  using TrackMergerCounterHost = PortableHostCollection<TrackMergerCounterSoA>;
}  // namespace reco
#endif  // RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterHost_h
