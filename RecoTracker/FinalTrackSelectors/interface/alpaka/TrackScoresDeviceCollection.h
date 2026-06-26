#ifndef RecoTracker_FinalTrackSelectors_alpaka_TrackScoresDeviceCollection_h
#define RecoTracker_FinalTrackSelectors_alpaka_TrackScoresDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackTorchClassifierFeaturesSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using TrackScoresDeviceCollection = PortableCollection<TrackTorchClassifierScoresSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_FinalTrackSelectors_alpaka_TrackScoresDeviceCollection_h
