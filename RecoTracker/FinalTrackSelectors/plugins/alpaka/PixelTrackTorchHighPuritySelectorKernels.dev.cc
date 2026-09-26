#include <alpaka/alpaka.hpp>
#include <xtd/math/sqrt.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"

#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackTorchHighPuritySelectorKernels.h"
#include "RecoTracker/PixelSeeding/interface/CATrackFeatures.h"
#include "RecoTracker/PixelSeeding/interface/OTHitTag.h"

//#define KERNELS_DEBUG

// ------------------------------------------------------------------------------

// Indices to the 5-dimensional track state vector (CMS convention)
static constexpr auto kStatePhi = 0;
static constexpr auto kStateDxy = 1;
static constexpr auto kStateCotTheta = 3;
static constexpr auto kStateDz = 4;

// Pixel-cluster feature block (columns 36-42) constants; they must match the nano producer the
// 42-feature forest is trained from. Modules are indexed in DetId order, so the Phase-2 pixel barrel
// occupies [0, 864) == phase2PixelTopology::layerStart[4]. Used only for the path-length
// normalisation of the cluster charge.
static constexpr uint32_t kPixelBarrelModuleEnd = 864;
// Path-length-normalised cluster charge (electrons) below which a pixel hit counts as low-charge.
static constexpr float kLowChargeThreshold = 7000.f;

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

  // ------------------------------------------------------------------------------

  // `iteration` is a column of the multi-iteration TrackSoA only; detect it so this translation
  // unit compiles against both layouts. Where the column is absent there is one iteration and the
  // feature is the constant 0 (== promptHighPt). The access must sit in a function template: on the
  // concrete element type the `if constexpr` condition is not value-dependent, so the discarded
  // branch would still have to compile.
  template <typename T, typename = void>
  struct HasIterationColumn : std::false_type {};
  template <typename T>
  struct HasIterationColumn<T, std::void_t<decltype(std::declval<T const&>().iteration())>> : std::true_type {};

  template <typename TTrack>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float trackIterationId(TTrack const& track) {
    if constexpr (HasIterationColumn<TTrack>::value)
      return float(static_cast<int>(track.iteration()));
    else
      return 0.f;
  }

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
      if (cms::alpakatools::once_per_grid(acc)) {
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
                                  const ::reco::TrackHitSoAConstView track_hits,
                                  const caStructures::CAHitsView hits,
                                  const int nHitsTot,
                                  const ::reco::OTRecHitsConstView otHits,  // raw OT-extra positions
                                  const uint32_t nOTHits,                   // 0 => merged-hits-only (view unused)
                                  const int* preselectedTrackIndices,
                                  const int* nPreselectedTracks,
                                  PixelTrackFitFeaturesView fitFeatures,
                                  PixelTrackHitFeaturesView hitFeatures,
                                  int* trackHitCounts) const {
      /**
            * Extracts per-track features used as input to
            * the Torch HighPurity classifier.
            *
            * For each valid preselected track:
            *  - Per-track features are written to the fit block, and to the hit block when
            *    that block is allocated
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
      // the hit block is allocated only when a model reads it: its extent is the flag
      const bool useHitFeatures = hitFeatures.metadata().size() > 0;
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
          fitFeatures.chi2(i) = track.chi2();  // in the SoA chi2 is stored as chi2/ndof
          fitFeatures.dzError(i) = xtd::sqrt(cov(kCovDzDz));
          fitFeatures.dxyError(i) = xtd::sqrt(cov(kCovDxyDxy));
          fitFeatures.eta(i) = track.eta();
          fitFeatures.nHits(i) = numHits;
          fitFeatures.phi(i) = state(kStatePhi);
          fitFeatures.phiError(i) = xtd::sqrt(cov(kCovPhiPhi));
          fitFeatures.pt(i) = track.pt();
          fitFeatures.qOverPtError(i) = xtd::sqrt(cov(kCovQOverPtQOverPt));
          fitFeatures.dzBS(i) = state(kStateDz);
          fitFeatures.dxyBS(i) = state(kStateDxy);
          fitFeatures.nLayers(i) = track.nLayers();
          fitFeatures.cotThetaError(i) = xtd::sqrt(cov(kCovCotThetaCotTheta));
          fitFeatures.covCotThetaDz(i) = cov(kCovCotThetaDz);
          fitFeatures.covDxyQOverPt(i) = cov(kCovDxyQOverPt);
          fitFeatures.covPhiDxy(i) = cov(kCovPhiDxy);
          fitFeatures.covPhiQOverPt(i) = cov(kCovPhiQOverPt);

          // The hit block (CA hit/stub, merged-collection provenance, pixel cluster) is filled
          // only for the hit-feature models. The CA features come from the shared
          // caTrackFeatures::fill, so the in-kernel gate, the nano table and this selector see the
          // same numbers. The hit walk indexes the merged TrackingRecHitsSoA over the per-track
          // [start,end) range of track_hits.id().
          if (useHitFeatures) {
            caTrackFeatures::Features feat;
            // Extras (hit cols 10-13): rzChi2 (r-z linearity), meanStubKappa, leverArm (rMax-r0),
            // rMax (radial extent), all produced in the same hit walk. Default-constructed to the
            // sentinels kept when fill() fails on a corrupt or short list.
            caTrackFeatures::Extras extras;
            const auto start = (inputTrackIdx == 0) ? 0u : tracks[inputTrackIdx - 1].hitOffsets();
            const auto end = track.hitOffsets();
            bool featOk = false;
            if (end > start && end <= (uint32_t)track_hits.metadata().size()) {
              // OT-aware: tagged extras resolve their position through the OT view (nullptr => none).
              const ::reco::OTRecHitsConstView* otViewPtr = (nOTHits > 0u) ? &otHits : nullptr;
              featOk = caTrackFeatures::fill(track_hits.id().data() + start,
                                             track_hits.id().data() + end,
                                             hits,
                                             nHitsTot,
                                             float(track.nLayers()),
                                             track.chi2(),
                                             feat,
                                             &extras,
                                             otViewPtr);
            }
            // CA features -> appended columns, dropping nHits and nLayers (redundant with the
            // columns of the same name). Order must match the trained model's input schema; a
            // failed fill() falls back to 0, as padding rows do.
            hitFeatures.caFitChi2(i) = featOk ? feat.caFitChi2 : 0.f;
            hitFeatures.psFrac(i) = featOk ? feat.psFrac : 0.f;
            hitFeatures.r0(i) = featOk ? feat.r0 : 0.f;
            hitFeatures.nPS(i) = featOk ? feat.nPS : 0.f;
            hitFeatures.spanZ(i) = featOk ? feat.spanZ : 0.f;
            hitFeatures.nStubs(i) = featOk ? feat.nStubs : 0.f;
            hitFeatures.logChi2Stub(i) = featOk ? feat.logChi2Stub : 0.f;
            hitFeatures.kErr(i) = featOk ? feat.kErr : 0.f;
            hitFeatures.dcaEst(i) = featOk ? feat.dcaEst : 0.f;
            hitFeatures.nBarrel(i) = featOk ? feat.nBarrel : 0.f;
            hitFeatures.rzChi2(i) = extras.rzChi2;
            hitFeatures.meanStubKappa(i) = extras.meanStubKappa;
            hitFeatures.leverArm(i) = featOk ? extras.leverArm : 0.f;
            hitFeatures.rMax(i) = featOk ? extras.rMax : 0.f;

            // Merged-collection provenance (cols 32-35: nAttached, nOTExtras, iterationId, ndof)
            // and the pixel-cluster charge/shape block (cols 36-42: minCharge, meanCharge,
            // minChargeNorm, maxSizeY, meanSizeY, maxSizeX, nLowCharge), in trained ABI order.
            // Gathered here rather than in caTrackFeatures::fill, which sees neither the per-hit
            // attached() flag nor the rechit cluster columns: one extra pass over the same
            // [start,end) span, under the same validity guard.
            int nAttachedHits = 0;
            int nOTExtraHits = 0;
            // Path-length normalisation of the cluster charge: state(kStateCotTheta) is
            // cot(theta), so |sin(theta)| = 1/sqrt(1+cot^2) and |cos(theta)| = |cot|/sqrt(1+cot^2).
            // A barrel sensor's normal is radial, so the path through it scales as 1/|sin(theta)|
            // -> normalised charge Q*|sin(theta)|; an endcap normal is along z -> Q*|cos(theta)|.
            // Same order of operations as the nano producer, so the two agree bit for bit.
            const float cotTheta = state(kStateCotTheta);
            const float invHyp = 1.f / std::sqrt(1.f + cotTheta * cotTheta);
            const float absSinTheta = invHyp;                       // barrel path factor
            const float absCosTheta = std::abs(cotTheta) * invHyp;  // endcap path factor
            int nPix = 0, nLow = 0;
            float qMin = 0.f, qSum = 0.f, qnMin = 0.f;
            float syMax = 0.f, sySum = 0.f, sxMax = 0.f;
            if (end > start && end <= (uint32_t)track_hits.metadata().size()) {
              // Outer-tracker entries start at offsetStubs and carry no cluster information; skip them.
              const uint32_t offsetStubsMain = hits.offsetStubs();
              for (uint32_t k = start; k < end; ++k) {
                const uint32_t h = track_hits[k].id();
                nAttachedHits += (track_hits[k].attached() == 1) ? 1 : 0;
                const bool otExtra = caOTHitTag::isOTId(h);
                nOTExtraHits += otExtra ? 1 : 0;
                // Cols 36-42 see pixel hits only, with the nano producer's selection.
                if (otExtra)
                  continue;  // raw OT extra: indexes the OT SoA, no cluster information there
                if (h >= (uint32_t)nHitsTot || h >= offsetStubsMain)
                  continue;  // stub row (no cluster information) or corrupt index
                auto const pix = hits.pixel(int32_t(h));
                const float q = float(pix.chargeAndStatus().charge);
                if (!(q > 0.f))
                  continue;  // a pixel row with no charge carries no usable cluster
                const bool barrel = (uint32_t)hits[h].detectorIndex() < kPixelBarrelModuleEnd;
                const float qn = q * (barrel ? absSinTheta : absCosTheta);
                // clusterSizeX/Y are raw signed 1/8-pixel sizes (clipped at 127, negated when the
                // cluster touches a sensor edge, see pixelCPEforDevice.h), used as stored: no abs()
                // and no /8, matching the nano table the model is trained from. syMax/sxMax start at
                // 0, so an all-edge track reports 0 rather than a negative maximum.
                const float sx = float(pix.clusterSizeX());
                const float sy = float(pix.clusterSizeY());
                if (nPix == 0) {
                  qMin = q;
                  qnMin = qn;
                } else {
                  qMin = std::min(qMin, q);
                  qnMin = std::min(qnMin, qn);
                }
                qSum += q;
                sySum += sy;
                syMax = std::max(syMax, sy);
                sxMax = std::max(sxMax, sx);
                if (qn < kLowChargeThreshold)
                  ++nLow;
                ++nPix;
              }
            }
            hitFeatures.nAttached(i) = float(nAttachedHits);
            hitFeatures.nOTExtras(i) = float(nOTExtraHits);
            // iteration exists only on the multi-iteration TrackSoA -> 0 where the column is absent.
            hitFeatures.iterationId(i) = trackIterationId(track);
            hitFeatures.ndof(i) = float(track.ndof());
            // Cluster columns: -1 sentinel on every column when the track carries no usable pixel
            // cluster, as in the nano producer.
            const bool clOk = (nPix > 0);
            hitFeatures.minCharge(i) = clOk ? qMin : -1.f;
            hitFeatures.meanCharge(i) = clOk ? qSum / float(nPix) : -1.f;
            hitFeatures.minChargeNorm(i) = clOk ? qnMin : -1.f;
            hitFeatures.maxSizeY(i) = clOk ? syMax : -1.f;
            hitFeatures.meanSizeY(i) = clOk ? sySum / float(nPix) : -1.f;
            hitFeatures.maxSizeX(i) = clOk ? sxMax : -1.f;
            hitFeatures.nLowCharge(i) = clOk ? float(nLow) : -1.f;
          }
        }
        // Case 2: padding entries --> fill with 0s for inference
        else {
          fitFeatures.chi2(i) = 0;
          fitFeatures.dzError(i) = 0;
          fitFeatures.dxyError(i) = 0;
          fitFeatures.eta(i) = 0;
          fitFeatures.nHits(i) = 0;
          fitFeatures.phi(i) = 0;
          fitFeatures.phiError(i) = 0;
          fitFeatures.pt(i) = 0;
          fitFeatures.qOverPtError(i) = 0;
          fitFeatures.dzBS(i) = 0;
          fitFeatures.dxyBS(i) = 0;
          fitFeatures.nLayers(i) = 0;
          fitFeatures.cotThetaError(i) = 0;
          fitFeatures.covCotThetaDz(i) = 0;
          fitFeatures.covDxyQOverPt(i) = 0;
          fitFeatures.covPhiDxy(i) = 0;
          fitFeatures.covPhiQOverPt(i) = 0;
          if (useHitFeatures) {
            hitFeatures.caFitChi2(i) = 0;
            hitFeatures.psFrac(i) = 0;
            hitFeatures.r0(i) = 0;
            hitFeatures.nPS(i) = 0;
            hitFeatures.spanZ(i) = 0;
            hitFeatures.nStubs(i) = 0;
            hitFeatures.logChi2Stub(i) = 0;
            hitFeatures.kErr(i) = 0;
            hitFeatures.dcaEst(i) = 0;
            hitFeatures.nBarrel(i) = 0;
            hitFeatures.rzChi2(i) = -1.f;
            hitFeatures.meanStubKappa(i) = 0;
            hitFeatures.leverArm(i) = 0;
            hitFeatures.rMax(i) = 0;
            hitFeatures.nAttached(i) = 0;
            hitFeatures.nOTExtras(i) = 0;
            hitFeatures.iterationId(i) = 0;
            hitFeatures.ndof(i) = 0;
            // The cluster columns pad with their -1 sentinel rather than 0. Padding rows are never scored.
            hitFeatures.minCharge(i) = -1.f;
            hitFeatures.meanCharge(i) = -1.f;
            hitFeatures.minChargeNorm(i) = -1.f;
            hitFeatures.maxSizeY(i) = -1.f;
            hitFeatures.meanSizeY(i) = -1.f;
            hitFeatures.maxSizeX(i) = -1.f;
            hitFeatures.nLowCharge(i) = -1.f;
          }
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
                                  ::reco::TrackHitSoAView track_hits_out,
                                  uint32_t* selectedCounts) const {
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
            *  - selectedCounts (optional) receives the two counts this kernel commits, so a host
            *    consumer can size against what was written instead of against the capacity
        */

      const auto nTracks = alpaka::math::min(acc, *nSelectedTracks, maxPreselectedTracks);
      if (cms::alpakatools::once_per_grid(acc))
        tracks_out.nTracks() = nTracks;

      // The output hit block is sized from an average (maxPreselectedTracks * avgHitsPerTrack)
      // while selectedTrackHitOffsets is the true running total, so no invariant relates the two.
      // Every write below is clamped to the block, and the CSR end offset to the same bound, so a
      // truncated copy still describes the hits it wrote.
      const uint32_t outHitCap = uint32_t(track_hits_out.metadata().size());

      // Counts committed by this kernel: [0] the tracks it writes, [1] the hits it writes (the CSR
      // end offset of the last written track, under the same clamp as the per-track offsets). Both
      // are exact upper bounds on the collection's content, so a consumer can allocate from them.
      if (selectedCounts != nullptr && cms::alpakatools::once_per_grid(acc)) {
        selectedCounts[0] = uint32_t(nTracks);
        const uint32_t hitEndTot = (nTracks > 0) ? uint32_t(selectedTrackHitOffsets[nTracks - 1]) : 0u;
        selectedCounts[1] = (hitEndTot < outHitCap) ? hitEndTot : outHitCap;
      }

      // Tail: the slots past the last track carry its CSR end offset, so nHits() reads zero there and a
      // reader that walks the SoA up to the first empty slot (the SoA monitors and comparisons) stops
      // at the right place instead of running into allocator garbage.
      {
        const uint32_t hitEndTot = (nTracks > 0) ? uint32_t(selectedTrackHitOffsets[nTracks - 1]) : 0u;
        const uint32_t hitEndTail = (hitEndTot < outHitCap) ? hitEndTot : outHitCap;
        for (auto k : cms::alpakatools::uniform_elements(acc, uint32_t(tracks_out.metadata().size())))
          if (k >= uint32_t(nTracks))
            tracks_out[k].hitOffsets() = hitEndTail;
      }

      for (auto i : cms::alpakatools::uniform_elements(acc, nTracks)) {
        const auto inputTrackIdx = selectedTrackIndices[i];
        if (inputTrackIdx >= 0) {
          const auto& track = tracks[inputTrackIdx];
          tracks_out[i] = track;
          const uint32_t hitEndOut = uint32_t(selectedTrackHitOffsets[i]);
          tracks_out[i].hitOffsets() = (hitEndOut < outHitCap) ? hitEndOut : outHitCap;

          //Access the hits associated to the track:
          auto hitBegin = (inputTrackIdx == 0) ? 0 : tracks[inputTrackIdx - 1].hitOffsets();
          auto hitEnd = track.hitOffsets();
          auto outStart = (i == 0) ? 0u : uint32_t(selectedTrackHitOffsets[i - 1]);

          const uint32_t nCopy = (hitEnd > hitBegin) ? uint32_t(hitEnd - hitBegin) : 0u;
          const uint32_t nRoom = (outStart < outHitCap) ? (outHitCap - outStart) : 0u;
          for (auto h = 0u; h < ((nCopy < nRoom) ? nCopy : nRoom); ++h) {
            track_hits_out[outStart + h].id() = track_hits[hitBegin + h].id();
            track_hits_out[outStart + h].detId() = track_hits[hitBegin + h].detId();
            track_hits_out[outStart + h].attached() = track_hits[hitBegin + h].attached();
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

  // Fit block: FeaturesExtractorKernel writes it for every preselected track, while the hit block
  // is written only under useHitFeatures, so the guard below covers the fit block alone rather than
  // reading memory that may never have been allocated.
  inline constexpr int kNForestFitFeatures = kNPixelTrackFitFeatures;

  // The fit-derived columns, one named float each; asArray() gives them in column order.
  struct ForestFitFeatures {
    float chi2 = 0.f;
    float dzError = 0.f;
    float dxyError = 0.f;
    float eta = 0.f;
    float nHits = 0.f;
    float phi = 0.f;
    float phiError = 0.f;
    float pt = 0.f;
    float qOverPtError = 0.f;
    float dzBS = 0.f;
    float dxyBS = 0.f;
    float nLayers = 0.f;
    float cotThetaError = 0.f;
    float covCotThetaDz = 0.f;
    float covDxyQOverPt = 0.f;
    float covPhiDxy = 0.f;
    float covPhiQOverPt = 0.f;

    ALPAKA_FN_HOST_ACC constexpr std::array<float, kNForestFitFeatures> asArray() const {
      return {chi2,
              dzError,
              dxyError,
              eta,
              nHits,
              phi,
              phiError,
              pt,
              qOverPtError,
              dzBS,
              dxyBS,
              nLayers,
              cotThetaError,
              covCotThetaDz,
              covDxyQOverPt,
              covPhiDxy,
              covPhiQOverPt};
    }
  };
  static_assert(std::is_standard_layout_v<ForestFitFeatures>);

  // The tree walk cannot report a bad input: the split test `f[k] < threshold` is false for a NaN,
  // so a non-finite feature takes the right branch at every node and the forest returns a finite,
  // meaningless score. The rejection is therefore made on the features, not on the score.
  // edm::isNotFinite is a bit-pattern test on the exponent field, so it survives -Ofast.
  template <typename TIdx>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool forestFitFeaturesFinite(const PixelTrackFitFeaturesConstView& fitFeatures,
                                                              const TIdx i) {
    ForestFitFeatures ff;
    ff.chi2 = fitFeatures[i].chi2();
    ff.dzError = fitFeatures[i].dzError();
    ff.dxyError = fitFeatures[i].dxyError();
    ff.eta = fitFeatures[i].eta();
    ff.nHits = fitFeatures[i].nHits();
    ff.phi = fitFeatures[i].phi();
    ff.phiError = fitFeatures[i].phiError();
    ff.pt = fitFeatures[i].pt();
    ff.qOverPtError = fitFeatures[i].qOverPtError();
    ff.dzBS = fitFeatures[i].dzBS();
    ff.dxyBS = fitFeatures[i].dxyBS();
    ff.nLayers = fitFeatures[i].nLayers();
    ff.cotThetaError = fitFeatures[i].cotThetaError();
    ff.covCotThetaDz = fitFeatures[i].covCotThetaDz();
    ff.covDxyQOverPt = fitFeatures[i].covDxyQOverPt();
    ff.covPhiDxy = fitFeatures[i].covPhiDxy();
    ff.covPhiQOverPt = fitFeatures[i].covPhiQOverPt();
    const auto f = ff.asArray();
    bool ok = true;
    for (int k = 0; ok && k < kNForestFitFeatures; ++k)
      ok = !edm::isNotFinite(f[k]);
    return ok;
  }

  // ------------------------------------------------------------------------------

  struct ScoreSelectionMaskKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const int maxPreselectedTracks,
                                  const double scoreThreshold,
                                  const double scoreThresholdLowDxy,
                                  const double dxyRampKnee,
                                  const PixelTrackFitFeaturesConstView fitFeatures,
                                  const int* nPreselectedTracks,
                                  const PixelTrackScoresSoA::View trackScores,
                                  int* selectionMask) const {
      /**
            * Applies a DNN score threshold to preselected tracks.
            *
            * For each track slot:
            *  - Reads the Torch score
            *  - Marks the track as selected if score >= threshold(|dxyBS|).
            *
            * dxy-aware threshold: the threshold ramps linearly from scoreThresholdLowDxy at
            * |dxyBS| = 0 down to scoreThreshold at |dxyBS| >= dxyRampKnee.
            * scoreThresholdLowDxy < 0 DISABLES the ramp (flat scoreThreshold everywhere), which is
            * the default and reproduces the plain threshold cut exactly.
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
        float thr = scoreThreshold;
        if (scoreThresholdLowDxy >= 0.) {
          const float adxy = alpaka::math::abs(acc, fitFeatures[i].dxyBS());
          const float ramp = (adxy >= float(dxyRampKnee)) ? 0.f : (1.f - adxy / float(dxyRampKnee));
          thr = float(scoreThreshold) + (float(scoreThresholdLowDxy) - float(scoreThreshold)) * ramp;
        }
        // A track whose fit produced a non-finite chi2, parameter or covariance-derived error is
        // rejected outright: the forest's verdict on it is meaningless. The score comparison is kept
        // in the promoting form (`score >= thr` -> keep), which leaves a non-finite score on the
        // rejecting side under -Ofast (-ffinite-math-only); the negated form is rewritable.
        const bool fitOk = forestFitFeaturesFinite(fitFeatures, i);
        selectionMask[i] = (fitOk && score >= thr) ? 1 : 0;
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
      if (cms::alpakatools::once_per_grid(acc)) {
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

    // preselectionMask must stay zeroed: PreselectionMaskingKernel writes only
    // [0, min(maxNumberOfTracks, nTracks)) while the scan below sweeps the whole capacity.
    // tmpPreselectedTrackIndices needs no fill: the compaction below loads element i only where
    // preselectionMask[i] == 1, which the zeroed mask confines to that same range.
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

    // Launch prefix-scan over the preselection mask to compute offsets.
    // iterativePrefixScan rather than multiBlockPrefixScan: the extent is the capacity while the
    // real track count is far smaller, and multiBlockPrefixScan finishes with a single-block tail
    // that read-modify-writes all `size` elements on one SM. Integer addition is exact and
    // associative, so both give the same inclusive scan.
    cms::alpakatools::iterativePrefixScan<Acc1D>(
        preselectionMask.data(), preselectionOffsets, uint32_t(maxNumberOfTracks), queue);

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
                               const ::reco::TrackHitSoAConstView track_hits,
                               const caStructures::CAHitsView hits,
                               const int nHitsTot,
                               const ::reco::OTRecHitsConstView otHits,
                               const uint32_t nOTHits,
                               const int* preselectedTrackIndices,
                               const int* nPreselectedTracks,
                               PixelTrackFitFeaturesView fitFeatures,
                               PixelTrackHitFeaturesView hitFeatures,
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
                        track_hits,
                        hits,
                        nHitsTot,
                        otHits,
                        nOTHits,
                        preselectedTrackIndices,
                        nPreselectedTracks,
                        fitFeatures,
                        hitFeatures,
                        trackHitCounts);
  }

  // ------------------------------------------------------------------------------

  void launchScoreFilter(Queue& queue,
                         const int maxPreselectedTracks,
                         const double scoreThreshold,
                         const double scoreThresholdLowDxy,
                         const double dxyRampKnee,
                         const PixelTrackFitFeaturesConstView fitFeatures,
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

    // selectionMask and selectedTrackHitCounts must stay zeroed: the kernels that fill them stop at
    // the track count while the scans that consume them sweep the whole capacity. selectionOffsets
    // is the scan's output, fully written before any read.
    alpaka::memset(queue, selectionMask, 0);
    alpaka::memset(queue, selectedTrackHitCounts, 0);

    constexpr auto threadsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        ScoreSelectionMaskKernel{},
                        maxPreselectedTracks,
                        scoreThreshold,
                        scoreThresholdLowDxy,
                        dxyRampKnee,
                        fitFeatures,
                        nPreselectedTracks,
                        trackScores,
                        selectionMask.data());

    // Apply the selection mask to compact the preselectedTrackIndices array
    // and produce the final list of selected tracks,
    // while also counting the number of kept tracks
    constexpr auto threadsPrefixScan = 256u;
    auto blocksPrefixScan = (maxPreselectedTracks + threadsPrefixScan - 1) / threadsPrefixScan;
    auto workDivPrefixScan = cms::alpakatools::make_workdiv<Acc1D>(blocksPrefixScan, threadsPrefixScan);

    // Launch prefix-scan over the selection mask to compute offsets (see launchCAPreselection for
    // why iterativePrefixScan).
    cms::alpakatools::iterativePrefixScan<Acc1D>(
        selectionMask.data(), selectionOffsets.data(), uint32_t(maxPreselectedTracks), queue);

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
    cms::alpakatools::iterativePrefixScan<Acc1D>(
        selectedTrackHitCounts.data(), selectedTrackHitOffsets, uint32_t(maxPreselectedTracks), queue);
  }

  // ------------------------------------------------------------------------------

  reco::TracksSoACollection launchProduceOutputTracks(Queue& queue,
                                                      const int maxPreselectedTracks,
                                                      const int avgHitsPerTrack,
                                                      const ::reco::TrackSoAConstView tracks,
                                                      const ::reco::TrackHitSoAConstView track_hits,
                                                      const int* selectedTrackIndices,
                                                      const int* nSelectedTracks,
                                                      const int* selectedTrackHitOffsets,
                                                      uint32_t* selectedCounts) {
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
                        tracks_out.view().trackHits(),
                        selectedCounts);

    return tracks_out;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
