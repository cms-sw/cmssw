#ifndef DataFormats_TrackSoA_test_alpaka_TrackSoAHeterogeneous_test_h
#define DataFormats_TrackSoA_test_alpaka_TrackSoAHeterogeneous_test_h

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testTrackSoA {

  void runKernels(::reco::TrackSoAView tracks_view, Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testTrackSoA

#endif  // DataFormats_TrackSoA_test_alpaka_TrackSoAHeterogeneous_test_h
