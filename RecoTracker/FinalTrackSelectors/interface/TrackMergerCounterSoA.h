#ifndef RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterSoA_h
#define RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterSoA_h

#include <Eigen/Core>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  // pair of indices
  GENERATE_SOA_LAYOUT(TrackMergerCounterLayout,
                      SOA_COLUMN(uint16_t, collection),
                      SOA_COLUMN(uint32_t, track),
                      SOA_COLUMN(uint32_t, hitsInTrack))

  using TrackMergerCounterSoA = TrackMergerCounterLayout<>;
  using TrackMergerCounterSoAView = TrackMergerCounterSoA::View;
  using TrackMergerCounterSoAConstView = TrackMergerCounterSoA::ConstView;

}  // namespace reco

#endif  // RecoTracker_FinalTrackSelectors_interface_TrackMergerCounterSoA_h
