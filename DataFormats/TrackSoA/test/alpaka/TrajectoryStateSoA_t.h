#ifndef DataFormats_TrackSoA_test_alpaka_TrajectoryStateSoA_t_h
#define DataFormats_TrackSoA_test_alpaka_TrajectoryStateSoA_t_h

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  template <typename TrackerTraits>
  void testTrackSoA(Queue& queue, ::reco::TrackSoAView& tracks);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test

#endif  // DataFormats_TrackSoA_test_alpaka_TrajectoryStateSoA_t_h
