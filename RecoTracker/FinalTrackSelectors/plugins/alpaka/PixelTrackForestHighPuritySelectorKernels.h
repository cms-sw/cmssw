#ifndef RecoTracker_FinalTrackSelectors_plugins_alpaka_PixelTrackForestHighPuritySelectorKernels_h
#define RecoTracker_FinalTrackSelectors_plugins_alpaka_PixelTrackForestHighPuritySelectorKernels_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void launchTreeScore(Queue& queue,
                       const int maxPreselectedTracks,
                       const int8_t* treeFeat,
                       const float* treeVal,
                       const int32_t* treeLeft,
                       const int32_t* treeRight,
                       const int32_t* treeRoots,
                       const int nTrees,
                       const float baseLogit,
                       const PixelTrackFeaturesSoA::ConstView trackFeatures,
                       const int* nPreselectedTracks,
                       PixelTrackScoresSoA::View trackScores);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_FinalTrackSelectors_plugins_alpaka_PixelTrackForestHighPuritySelectorKernels_h
