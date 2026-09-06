#ifndef RecoTracker_FinalTrackSelectors_plugins_alpaka_PixelTrackTorchHighPuritySelectorKernels_h
#define RecoTracker_FinalTrackSelectors_plugins_alpaka_PixelTrackTorchHighPuritySelectorKernels_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void launchCAPreselection(Queue& queue,
                            const int maxNumberOfTracks,
                            const int minNumberOfHits,
                            const ::pixelTrack::Quality minimumTrackQuality,
                            const ::reco::TrackSoAConstView tracks,
                            int* preselectedTrackIndices,
                            int* preselectionOffsets,
                            int* nPreselectedTracks);

  void launchFeaturesExtractor(Queue& queue,
                               const int maxPreselectedTracks,
                               const ::reco::TrackSoAConstView tracks,
                               const ::reco::TrackHitSoAConstView track_hits,
                               const ::reco::TrackingRecHitConstView hits,
                               const int nHitsTot,
                               // Raw OT-rechit view + count so the feature walk resolves tagged
                               // OT extras (nOTHits == 0 / empty view => merged-hits-only).
                               const ::reco::OTRecHitsConstView otHits,
                               const uint32_t nOTHits,
                               const bool useHitFeatures,
                               const int* preselectedTrackIndices,
                               const int* nPreselectedTracks,
                               PixelTrackFeaturesSoA::View trackFeatures,
                               int* trackHitCounts);

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

  void launchScoreFilter(Queue& queue,
                         const int maxPreselectedTracks,
                         const double scoreThreshold,
                         const double scoreThresholdLowDxy,
                         const double dxyRampKnee,
                         const PixelTrackFeaturesSoA::ConstView trackFeatures,
                         const PixelTrackScoresSoA::View trackScores,
                         const int* preselectedTrackIndices,
                         const int* nPreselectedTracks,
                         const int* trackHitCounts,
                         int* selectedTrackIndices,
                         int* nSelectedTracks,
                         int* selectedTrackHitOffsets);

  // selectedCounts (optional, may be null): 2-word device buffer the compaction kernel fills with
  // [0] tracks written into the output SoA, [1] hits written into the output hit block. Both are exact,
  // so a consumer can size against them.
  reco::TracksSoACollection launchProduceOutputTracks(Queue& queue,
                                                      const int maxPreselectedTracks,
                                                      const int avgHitsPerTrack,
                                                      const ::reco::TrackSoAConstView tracks,
                                                      const ::reco::TrackHitSoAConstView track_hits,
                                                      const int* selectedTrackIndices,
                                                      const int* nSelectedTracks,
                                                      const int* selectedTrackHitOffsets,
                                                      uint32_t* selectedCounts = nullptr);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_FinalTrackSelectors_plugins_alpaka_PixelTrackTorchHighPuritySelectorKernels_h
