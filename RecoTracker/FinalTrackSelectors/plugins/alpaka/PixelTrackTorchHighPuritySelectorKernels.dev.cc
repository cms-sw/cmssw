#include <alpaka/alpaka.hpp>
#include <xtd/math/sqrt.h>
#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"

#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackTorchHighPuritySelectorKernels.h"

//#define KERNELS_DEBUG

// ------------------------------------------------------------------------------

// Indices to the 5-dimensional track state vector (CMS convention)
static constexpr auto kStatePhi = 0;
static constexpr auto kStateDxy = 1;
static constexpr auto kStateDz = 4;

// Indices into the 5x5 track covariance matrix (CMS convention)
static constexpr auto kCovPhiPhi = 0;             // (0,0)
static constexpr auto kCovPhiDxy = 1;             // (0,1)
static constexpr auto kCovPhiQOverPt = 2;         // (0,2)
static constexpr auto kCovDxyDxy = 5;             // (1,1)
static constexpr auto kCovDxyQOverPt = 6;         // (1,2)
static constexpr auto kCovQOverPtQOverPt = 9;     // (2,2)
static constexpr auto kCovCotThetaCotTheta = 12;  // (3,3)
static constexpr auto kCovCotThetaDz = 13;        // (3,4)
static constexpr auto kCovDzDz = 14;              // (4,4)

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PixelTrackFeaturesSoAView = PixelTrackFeaturesSoA::View;
  using TrackHitSoA = ::reco::TrackHitSoA;

  // ------------------------------------------------------------------------------
  // --------------------------- Definitions of Kernels ---------------------------
  // ------------------------------------------------------------------------------

  struct PreselectionMaskingKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const int maxNumberOfTracks,
                                  const int minNumberOfHits,
                                  const ::pixelTrack::Quality minimumTrackQuality,
                                  const ::reco::TrackSoAConstView tracks,
                                  int* preselectionMask,
                                  int* tmpPreselectedTrackIndices) const {
      /**
            * Applies a fast preselection to pixel tracks based on:
            *  - CAHitNtuplet quality flag
            *  - minimum number of associated hits
            *
            * Inputs:
            *  - tracks              : input TrackSoA
            *  - maxNumberOfTracks   : maximum number of tracks to consider
            *  - minNumberOfHits     : minimum number of hits per track
            *  - minimumTrackQuality : minimum allowed track quality
            *
            * Outputs:
            *  - preselectionMask[i] = 1 if track i passes preselection, 0 otherwise
            *  - tmpPreselectedTrackIndices[i] = i (identity mapping, used for compaction)
            *
            * Notes:
            *  - Only tracks in [0, min(maxNumberOfTracks, tracks.nTracks())) are processed
            *  - Entries beyond this range are left unchanged and are expected to be
            *    pre-initialised by the caller.
            *  - This kernel does not perform compaction; it only prepares the mask
        */

      const auto trackLimit = alpaka::math::min(acc, maxNumberOfTracks, tracks.nTracks());
#ifdef KERNELS_DEBUG
      if (cms::alpakatools::once_per_block(acc)) {
        printf("nTracks=%d\n", tracks.nTracks());
        if (tracks.nTracks() >= maxNumberOfTracks)
          printf("PixelTrackTorchHighPuritySelectorKernels Warning: nTracks (%d) >= maxNumberOfTracks (%d)\n",
                 tracks.nTracks(),
                 maxNumberOfTracks);
      }
#endif
      for (auto i : cms::alpakatools::uniform_elements(acc, trackLimit)) {
        tmpPreselectedTrackIndices[i] = i;
        bool isGoodQuality = tracks[i].quality() >= minimumTrackQuality && nHits(tracks, i) >= minNumberOfHits;
        preselectionMask[i] = isGoodQuality ? 1 : 0;
      }
    }
  };

  // ------------------------------------------------------------------------------

  struct FeaturesExtractorKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const int maxPreselectedTracks,
                                  const ::reco::TrackSoAConstView tracks,
                                  const int* preselectedTrackIndices,
                                  const int* nPreselectedTracks,
                                  PixelTrackFeaturesSoAView trackFeatures,
                                  int* trackHitCounts) const {
      /**
            * Extracts per-track features used as input to
            * the Torch HighPurity classifier.
            *
            * For each valid preselected track:
            *  - Per-track features are written to PixelTrackFeaturesSoA
            *  - trackHitCounts[i] stores the number of hits per track
            *    and is later transformed into hit offsets via prefix-scan

            *
            * Padding policy:
            *  - Slots i >= nPreselectedTracks are treated as padding
            *  - All padding slots are filled with 0s
            *
            * Preconditions:
            *  - preselectedTrackIndices contains a compact list of valid track indices
            *  - The first nPreselectedTracks entries are valid
            * This guarantees fixed-size tensors for Torch inference.
        */
      const auto nPreselected = *nPreselectedTracks;
      const auto nPreselectedTracksBound = alpaka::math::min(acc, nPreselected, maxPreselectedTracks);

      for (auto i : cms::alpakatools::uniform_elements(acc, maxPreselectedTracks)) {
        // Case 1: valid preselected track --> extract features

        if (i < (uint32_t)nPreselectedTracksBound) {
          auto inputTrackIdx = preselectedTrackIndices[i];
#ifdef KERNELS_DEBUG
          if (inputTrackIdx < 0)
            printf(
                "PixelTrackTorchHighPuritySelectorKernels: Invalid preselectedTrackIndices for preselected "
                "inputTrackIdx %d\n",
                i);
#endif
          // Access the track
          const auto& track = tracks[inputTrackIdx];
          const auto& cov = track.covariance();
          const auto& state = track.state();
          const auto numHits = nHits(tracks, inputTrackIdx);
          trackHitCounts[i] = numHits;

          // Fill per-track features
          trackFeatures.chi2(i) = track.chi2();  // in the SoA chi2 is stored as chi2/ndof
          trackFeatures.dzError(i) = xtd::sqrt(cov(kCovDzDz));
          trackFeatures.dxyError(i) = xtd::sqrt(cov(kCovDxyDxy));
          trackFeatures.eta(i) = track.eta();
          trackFeatures.nHits(i) = numHits;
          trackFeatures.phi(i) = state(kStatePhi);
          trackFeatures.phiError(i) = xtd::sqrt(cov(kCovPhiPhi));
          trackFeatures.pt(i) = track.pt();
          trackFeatures.qOverPtError(i) = xtd::sqrt(cov(kCovQOverPtQOverPt));
          trackFeatures.dzBS(i) = state(kStateDz);
          trackFeatures.dxyBS(i) = state(kStateDxy);
          trackFeatures.nLayers(i) = track.nLayers();
          trackFeatures.cotThetaError(i) = xtd::sqrt(cov(kCovCotThetaCotTheta));
          trackFeatures.covCotThetaDz(i) = cov(kCovCotThetaDz);
          trackFeatures.covDxyQOverPt(i) = cov(kCovDxyQOverPt);
          trackFeatures.covPhiDxy(i) = cov(kCovPhiDxy);
          trackFeatures.covPhiQOverPt(i) = cov(kCovPhiQOverPt);
        }
        // Case 2: padding entries --> fill with 0s for inference
        else {
          trackFeatures.chi2(i) = 0;
          trackFeatures.dzError(i) = 0;
          trackFeatures.dxyError(i) = 0;
          trackFeatures.eta(i) = 0;
          trackFeatures.nHits(i) = 0;
          trackFeatures.phi(i) = 0;
          trackFeatures.phiError(i) = 0;
          trackFeatures.pt(i) = 0;
          trackFeatures.qOverPtError(i) = 0;
          trackFeatures.dzBS(i) = 0;
          trackFeatures.dxyBS(i) = 0;
          trackFeatures.nLayers(i) = 0;
          trackFeatures.cotThetaError(i) = 0;
          trackFeatures.covCotThetaDz(i) = 0;
          trackFeatures.covDxyQOverPt(i) = 0;
          trackFeatures.covPhiDxy(i) = 0;
          trackFeatures.covPhiQOverPt(i) = 0;
        }
      }
    }
  };

  // ------------------------------------------------------------------------------

  struct PixelTrackFilterKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const int maxPreselectedTracks,
                                  const ::reco::TrackSoAConstView tracks,
                                  const ::reco::TrackHitSoAConstView track_hits,
                                  const int* selectedTrackIndices,
                                  const int* nSelectedTracks,
                                  const int* selectedTrackHitOffsets,
                                  ::reco::TrackSoAView tracks_out,
                                  ::reco::TrackHitSoAView track_hits_out) const {
      /**
            * Produces the final output TrackSoA by:
            *  - Copying selected tracks from the input TrackSoA
            *  - Copying and compacting the associated TrackHitSoA
            *
            * Inputs:
            *  - selectedTrackIndices[]: compact list of selected input track indices
            *  - nSelectedTracks: number of selected tracks
            *  - selectedTrackHitOffsets[]: inclusive prefix sum of per-track hit counts.
            *                 selectedTrackHitOffsets[i] stores the end offset of hits for track i.
            *
            * Outputs:
            *  - tracks_out           : compact TrackSoA containing selected tracks
            *  - track_hits_out       : compact TrackHitSoA containing selected hits
            *
            * Notes:
            *  - tracks_out.nTracks() is set by a single thread
            *  - Hit offsets in tracks_out are taken from selectedTrackHitOffsets[]
        */

      const auto nTracks = alpaka::math::min(acc, *nSelectedTracks, maxPreselectedTracks);
      if (cms::alpakatools::once_per_block(acc))
        tracks_out.nTracks() = nTracks;

      for (auto i : cms::alpakatools::uniform_elements(acc, nTracks)) {
        const auto inputTrackIdx = selectedTrackIndices[i];
        if (inputTrackIdx >= 0) {
          const auto& track = tracks[inputTrackIdx];
          tracks_out[i] = track;
          tracks_out[i].hitOffsets() = selectedTrackHitOffsets[i];

          //Access the hits associated to the track:
          auto hitBegin = (inputTrackIdx == 0) ? 0 : tracks[inputTrackIdx - 1].hitOffsets();
          auto hitEnd = track.hitOffsets();
          auto outStart = (i == 0) ? 0 : selectedTrackHitOffsets[i - 1];

          for (auto h = 0u; h < (hitEnd - hitBegin); ++h) {
            track_hits_out[outStart + h].id() = track_hits[hitBegin + h].id();
            track_hits_out[outStart + h].detId() = track_hits[hitBegin + h].detId();
          }
        } else {
#ifdef KERNELS_DEBUG
          printf("PixelTrackTorchHighPuritySelectorKernels: Error inputTrackIdx is negative");
#endif
        }
      }
    }
  };

  // ------------------------------------------------------------------------------

  struct ScoreSelectionMaskKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const int maxPreselectedTracks,
                                  const double scoreThreshold,
                                  const int* nPreselectedTracks,
                                  const PixelTrackScoresSoA::View trackScores,
                                  int* selectionMask) const {
      /**
            * Applies a DNN score threshold to preselected tracks.
            *
            * For each track slot:
            *  - Reads the Torch score
            *  - Marks the track as selected if:
            *      score >= scoreThreshold
            *
            * Outputs:
            *  - selectionMask[i] = 1 if track is selected, 0 otherwise
            *
            * Notes:
            *  - No compaction is performed in this kernel
        */
      const auto nPreselected = *nPreselectedTracks;
      const auto nValid = alpaka::math::min(acc, nPreselected, maxPreselectedTracks);
      for (auto i : cms::alpakatools::uniform_elements(acc, nValid)) {
        const auto score = trackScores[i].score();
        selectionMask[i] = (score >= scoreThreshold) ? 1 : 0;
      }
    }
  };

  // ------------------------------------------------------------------------------

  struct FilterArray {
    template <typename TAcc, typename T, typename Index, typename Size>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const T* __restrict__ old_array,
                                  T* __restrict__ new_array,
                                  const Index* __restrict__ offsets,
                                  Size old_size,
                                  Size* __restrict__ new_size) const {
      /**
                * Compacts an input array using precomputed inclusive prefix-sum offsets.
                *
                * Inputs:
                *  - old_array[] : input array
                *  - offsets[]   : inclusive prefix sum of a selection mask
                *  - old_size    : size of the input array
                *
                * Outputs:
                *  - new_array[] : compacted array
                *  - new_size    : total number of selected elements
                *
                * Notes:
                *  - offsets[last] defines the size of the compacted array
                *  - Only the first occurrence of each offset value writes to new_array
            */

      // ---- Compute output size once ----
      if (cms::alpakatools::once_per_block(acc)) {
        if (old_size > 0) {
          *new_size = static_cast<Size>(offsets[old_size - 1]);
        } else {
          *new_size = 0;
        }
      }

      // ---- Compaction ----
      for (auto i : cms::alpakatools::uniform_elements(acc, old_size)) {
        const auto off = offsets[i];
        const auto prev_off = (i == 0) ? 0 : offsets[i - 1];

        if (off != prev_off) {
          new_array[off - 1] = old_array[i];
        }
      }
    }
  };

  // ------------------------------------------------------------------------------
  // -------------------------- Definitions of Launchers --------------------------
  // ------------------------------------------------------------------------------

  void launchCAPreselection(Queue& queue,
                            const int maxNumberOfTracks,
                            const int minNumberOfHits,
                            const ::pixelTrack::Quality minimumTrackQuality,
                            const ::reco::TrackSoAConstView tracks,
                            int* preselectedTrackIndices,
                            int* preselectionOffsets,
                            int* nPreselectedTracks) {
    // Produce a preselection mask based on track quality and number of hits
    auto tmpPreselectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks);
    auto preselectionMask = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks);

    alpaka::memset(queue, tmpPreselectedTrackIndices, 0);
    alpaka::memset(queue, preselectionMask, 0);

    constexpr auto threadsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(maxNumberOfTracks, threadsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        PreselectionMaskingKernel{},
                        maxNumberOfTracks,
                        minNumberOfHits,
                        minimumTrackQuality,
                        tracks,
                        preselectionMask.data(),
                        tmpPreselectedTrackIndices.data());

    // Apply the preselection mask to compact the preselectedTrackIndices array
    // and produce the final list of preselected tracks,
    // while also counting the number of selected tracks
    constexpr auto threadsPrefixScan = 256u;
    auto blocksPrefixScan = (maxNumberOfTracks + threadsPrefixScan - 1) / threadsPrefixScan;
    auto workDivPrefixScan = cms::alpakatools::make_workdiv<Acc1D>(blocksPrefixScan, threadsPrefixScan);
    auto bCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, bCounter, 0);

    // Launch prefix-scan over the preselection mask to compute offsets
    alpaka::exec<Acc1D>(queue,
                        workDivPrefixScan,
                        cms::alpakatools::multiBlockPrefixScan<int>(),
                        preselectionMask.data(),
                        preselectionOffsets,
                        maxNumberOfTracks,
                        blocksPrefixScan,
                        bCounter.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    // Compact the preselectedTrackIndices array using the preselection offsets
    alpaka::exec<Acc1D>(queue,
                        workDivPrefixScan,
                        FilterArray{},
                        tmpPreselectedTrackIndices.data(),
                        preselectedTrackIndices,
                        preselectionOffsets,
                        maxNumberOfTracks,
                        nPreselectedTracks);
  }

  // ------------------------------------------------------------------------------

  void launchFeaturesExtractor(Queue& queue,
                               const int maxPreselectedTracks,
                               const ::reco::TrackSoAConstView tracks,
                               const int* preselectedTrackIndices,
                               const int* nPreselectedTracks,
                               PixelTrackFeaturesSoAView trackFeatures,
                               int* trackHitCounts) {
    // Extract per-track features for Torch inference
    constexpr auto threadsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        FeaturesExtractorKernel{},
                        maxPreselectedTracks,
                        tracks,
                        preselectedTrackIndices,
                        nPreselectedTracks,
                        trackFeatures,
                        trackHitCounts);
  }

  // ------------------------------------------------------------------------------

  void launchScoreFilter(Queue& queue,
                         const int maxPreselectedTracks,
                         const double scoreThreshold,
                         const PixelTrackScoresSoA::View trackScores,
                         const int* preselectedTrackIndices,
                         const int* nPreselectedTracks,
                         const int* trackHitCounts,
                         int* selectedTrackIndices,
                         int* nSelectedTracks,
                         int* selectedTrackHitOffsets) {
    // Produce a selection mask out of the DNN scores
    auto selectionMask = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks);
    auto selectionOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks);
    auto selectedTrackHitCounts = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks);

    alpaka::memset(queue, selectionMask, 0);
    alpaka::memset(queue, selectionOffsets, 0);
    alpaka::memset(queue, selectedTrackHitCounts, 0);

    constexpr auto threadsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        ScoreSelectionMaskKernel{},
                        maxPreselectedTracks,
                        scoreThreshold,
                        nPreselectedTracks,
                        trackScores,
                        selectionMask.data());

    // Apply the selection mask to compact the preselectedTrackIndices array
    // and produce the final list of selected tracks,
    // while also counting the number of kept tracks
    constexpr auto threadsPrefixScan = 256u;
    auto blocksPrefixScan = (maxPreselectedTracks + threadsPrefixScan - 1) / threadsPrefixScan;
    auto workDivPrefixScan = cms::alpakatools::make_workdiv<Acc1D>(blocksPrefixScan, threadsPrefixScan);
    auto bCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, bCounter, 0);

    // Launch prefix-scan over the selection mask to compute offsets
    alpaka::exec<Acc1D>(queue,
                        workDivPrefixScan,
                        cms::alpakatools::multiBlockPrefixScan<int>(),
                        selectionMask.data(),
                        selectionOffsets.data(),
                        maxPreselectedTracks,
                        blocksPrefixScan,
                        bCounter.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    // Compact the preselectedTrackIndices array using the selection offsets
    alpaka::exec<Acc1D>(queue,
                        workDivPrefixScan,
                        FilterArray{},
                        preselectedTrackIndices,
                        selectedTrackIndices,
                        selectionOffsets.data(),
                        maxPreselectedTracks,
                        nSelectedTracks);

    // Compact selectedTrackHitCounts using the same selection offsets to produce selectedTrackHitOffsets
    alpaka::exec<Acc1D>(queue,
                        workDivPrefixScan,
                        FilterArray{},
                        trackHitCounts,
                        selectedTrackHitCounts.data(),
                        selectionOffsets.data(),
                        maxPreselectedTracks,
                        nSelectedTracks);

    // Finally, compute the prefix-scan to get hit offsets
    alpaka::memset(queue, bCounter, 0);
    alpaka::exec<Acc1D>(queue,
                        workDivPrefixScan,
                        cms::alpakatools::multiBlockPrefixScan<int>(),
                        selectedTrackHitCounts.data(),
                        selectedTrackHitOffsets,
                        maxPreselectedTracks,
                        blocksPrefixScan,
                        bCounter.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
  }

  // ------------------------------------------------------------------------------

  reco::TracksSoACollection launchProduceOutputTracks(Queue& queue,
                                                      const int maxPreselectedTracks,
                                                      const int avgHitsPerTrack,
                                                      const ::reco::TrackSoAConstView tracks,
                                                      const ::reco::TrackHitSoAConstView track_hits,
                                                      const int* selectedTrackIndices,
                                                      const int* nSelectedTracks,
                                                      const int* selectedTrackHitOffsets) {
    reco::TracksSoACollection tracks_out(queue, int(maxPreselectedTracks), int(maxPreselectedTracks * avgHitsPerTrack));

    constexpr auto threadsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        PixelTrackFilterKernel{},
                        maxPreselectedTracks,
                        tracks,
                        track_hits,
                        selectedTrackIndices,
                        nSelectedTracks,
                        selectedTrackHitOffsets,
                        tracks_out.view().tracks(),
                        tracks_out.view().trackHits());

    return tracks_out;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
