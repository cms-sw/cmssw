#ifndef RecoTracker_FinalTrackSelectors_plugins_alpaka_TrackSoAMergerKernelsImpl_h
#define RecoTracker_FinalTrackSelectors_plugins_alpaka_TrackSoAMergerKernelsImpl_h

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
// #define CA_DEBUG
// #define CA_WARNINGS

// C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

// Alpaka includes
#include <alpaka/alpaka.hpp>

// CMSSW includes
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackMergerCounterSoA.h"

// local includes
#include "TrackSoAMergerKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::trackSoAMergerKernels {

  using namespace cms::alpakatools;

  class Kernel_fillGoodTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::mergerKernels::InputTracks const allTracks,
                                  ::reco::TrackMergerCounterSoAView cn,
                                  ::reco::TrackSoAView outTracks,
                                  ::reco::TrackHitSoAView outHits) const {
      auto nGoodTracks = outTracks.metadata().size();
#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("Kernel_fillGoodTracks: nGoodTracks: %u\n", nGoodTracks);
      }
#endif
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nGoodTracks)) {
        auto const collectionIndex = cn[t].collection();
        auto const inputTrackIndex = cn[t].track();

        auto const& inTracks = allTracks.views[collectionIndex];
        auto const& inHits = allTracks.hitViews[collectionIndex];

        outTracks[t].quality() = inTracks[inputTrackIndex].quality();
        outTracks[t].chi2() = inTracks[inputTrackIndex].chi2();
        outTracks[t].nLayers() = inTracks[inputTrackIndex].nLayers();
        outTracks[t].eta() = inTracks[inputTrackIndex].eta();
        outTracks[t].pt() = inTracks[inputTrackIndex].pt();
        outTracks[t].iteration() = inTracks[inputTrackIndex].iteration();
#ifdef GPU_DEBUG
        printf(
            "Track %u: collectionIndex: %u, inputTrackIndex: %u, quality: %d, chi2: %f, nLayers: %d, eta: %f, pt: %f "
            "\n",
            t,
            collectionIndex,
            inputTrackIndex,
            uint32_t(outTracks[t].quality()),
            outTracks[t].chi2(),
            outTracks[t].nLayers(),
            outTracks[t].eta(),
            outTracks[t].pt());
#endif
        for (uint32_t i = 0; i < 5; ++i)
          outTracks[t].state()(i) = inTracks[inputTrackIndex].state()(i);

        for (uint32_t i = 0; i < 15; ++i)
          outTracks[t].covariance()(i) = inTracks[inputTrackIndex].covariance()(i);

        ALPAKA_ASSERT_ACC((t == 0 && cn[t].hitsInTrack() == outTracks[t].hitOffsets()) ||
                          (t > 0 && cn[t].hitsInTrack() == outTracks[t].hitOffsets() - outTracks[t - 1].hitOffsets()));

        uint32_t const inHitBegin = inputTrackIndex == 0 ? 0 : inTracks[inputTrackIndex - 1].hitOffsets();
        uint32_t const inHitEnd = inTracks[inputTrackIndex].hitOffsets();

        uint32_t const outHitBegin = t == 0 ? 0 : outTracks[t - 1].hitOffsets();
        uint32_t const outHitEnd = outTracks[t].hitOffsets();
#ifdef GPU_DEBUG
        printf("Track %u: inHits: [%u, %u), outHits: [%u, %u)\n", t, inHitBegin, inHitEnd, outHitBegin, outHitEnd);
#endif
        ALPAKA_ASSERT_ACC(outHitEnd - outHitBegin == inHitEnd - inHitBegin);
#ifdef GPU_DEBUG
        printf("Track %u: inHits: [%u, %u), outHits: [%u, %u)\n", t, inHitBegin, inHitEnd, outHitBegin, outHitEnd);
#endif
        for (uint32_t h = 0; h < inHitEnd - inHitBegin; ++h) {
          outHits[outHitBegin + h].id() = inHits[inHitBegin + h].id();
          outHits[outHitBegin + h].detId() = inHits[inHitBegin + h].detId();
        }
      }

      if (cms::alpakatools::once_per_grid(acc))
        outTracks.nTracks() = nGoodTracks;
    }
  };

  class Kernel_countGoodTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::mergerKernels::InputTracks const allTracks,
                                  const pixelTrack::Quality minQuality,
                                  ::reco::TrackMergerCounterSoAView cn,
                                  uint32_t* totTracks,
                                  uint32_t* totHits) const {
#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc))
        printf("Kernel_countGoodTracks: nInputs: %u\n", allTracks.nInputs);
#endif

      for (uint32_t globalIndex : cms::alpakatools::uniform_elements(acc, allTracks.nTracks)) {
#ifdef GPU_DEBUG
        if (globalIndex % 5000 == 0)
          printf("globalIndex: %u %u - nInputs: %u\n", globalIndex, allTracks.nTracks, allTracks.nInputs);
#endif
        uint32_t trackIndex = globalIndex;

        for (int collectionIndex = 0; collectionIndex < allTracks.nInputs; ++collectionIndex) {
          auto const& tracks = allTracks.views[collectionIndex];
          uint32_t const capacity = tracks.metadata().size();

          if (trackIndex >= capacity) {
            trackIndex -= capacity;
            continue;
          }

          if (trackIndex >= uint32_t(tracks.nTracks()))
            continue;

          if (tracks[trackIndex].quality() >= minQuality) {
            auto t = alpaka::atomicAdd(acc, totTracks, 1u, alpaka::hierarchy::Blocks{});
            uint32_t h = ::reco::nHits(tracks, trackIndex);
            alpaka::atomicAdd(acc, totHits, h, alpaka::hierarchy::Blocks{});
#ifdef GPU_DEBUG
            if (trackIndex % 100 == 0)
              printf("collectionIndex: %d, trackIndex: %d, t: %d, h: %d\n", collectionIndex, trackIndex, t, h);
#endif
            cn[t].collection() = collectionIndex;
            cn[t].track() = trackIndex;
            cn[t].hitsInTrack() = h;
          }
          break;
        }
      }
    }
  };

  class Kernel_sameHitsDuplicates {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ::reco::TrackSoAView track_view,
                                  ::reco::TrackHitSoAView trackHit_view,
                                  const double matchFraction,
                                  const int minHitsForDuplicate) const {
#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("Kernel_sameHitsDuplicates: nTracks: %u\n", track_view.nTracks());
      }
#endif
      for (uint32_t i : cms::alpakatools::uniform_elements_x(acc, track_view.nTracks())) {
        auto const nHitsI = uint32_t(::reco::nHits(track_view, i));
        auto const hitBeginI = i == 0 ? 0 : track_view[i - 1].hitOffsets();

        for (uint32_t j : cms::alpakatools::uniform_elements_y(acc, i + 1, track_view.nTracks())) {
          if (nHitsI != uint32_t(::reco::nHits(track_view, j)))
            continue;

          auto const hitBeginJ = track_view[j - 1].hitOffsets();  // j > 0
          int matchedHits = 0;

          for (uint32_t k = 0; k < nHitsI; ++k) {
            if (trackHit_view[hitBeginI + k].id() == trackHit_view[hitBeginJ + k].id()) {
              ++matchedHits;
            }
          }

          if (double(matchedHits) / double(nHitsI) > matchFraction and matchedHits >= minHitsForDuplicate) {
            track_view[i].quality() = pixelTrack::Quality::dup;
#ifdef GPU_DEBUG
            printf(
                "Kernel_sameHitsDuplicates: i: %u, j: %u, matchedHits: %u, nHitsI: %u, matchFraction: %f, quality: "
                "%d\n",
                i,
                j,
                matchedHits,
                nHitsI,
                matchFraction,
                uint32_t(track_view[i].quality()));
#endif
            break;
          }
        }
      }
    }
  };

  class Kernel_trackParameterDuplicates {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ::reco::TrackSoAView track_view,
                                  const float nSigma2,
                                  const float maxDeltaR2,
                                  const float maxRelativePtDifference) const {
#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("Kernel_trackParameterDuplicates: nTracks: %u\n", track_view.nTracks());
      }
#endif
      for (uint32_t i : cms::alpakatools::uniform_elements_x(acc, track_view.nTracks())) {
        auto const qi = track_view[i].quality();

        if (qi == pixelTrack::Quality::dup)
          continue;

        auto const pti = track_view[i].pt();
        auto const etai = track_view[i].eta();
        auto const phii = ::reco::phi(track_view, i);
        auto const chargei = ::reco::charge(track_view, i);

        for (uint32_t j : cms::alpakatools::uniform_elements_y(acc, i + 1, track_view.nTracks())) {
          auto const qj = track_view[j].quality();
          if (qj == pixelTrack::Quality::dup)
            return;
          if (qi == pixelTrack::Quality::dup)
            break;

          if (chargei != ::reco::charge(track_view, j))
            return;

          auto deltaPhi = phii - ::reco::phi(track_view, j);
          if (deltaPhi > M_PI)
            deltaPhi -= 2.f * M_PI;
          else if (deltaPhi < -M_PI)
            deltaPhi += 2.f * M_PI;

          auto square = [](auto x) { return x * x; };

          if (square(deltaPhi) + square(etai - track_view[j].eta()) > maxDeltaR2)
            return;

          auto better = [&](uint32_t a, uint32_t b) {
            if (track_view[a].nLayers() != track_view[b].nLayers())
              return track_view[a].nLayers() > track_view[b].nLayers();
            if (track_view[a].quality() != track_view[b].quality())
              return track_view[a].quality() > track_view[b].quality();
            if (track_view[a].chi2() != track_view[b].chi2())
              return track_view[a].chi2() < track_view[b].chi2();
            return a < b;
          };

          auto const ptj = track_view[j].pt();
          auto const minPt = alpaka::math::min(acc, pti, ptj);
          if (minPt <= 0.f or alpaka::math::abs(acc, pti - ptj) > maxRelativePtDifference * minPt)
            return;

          constexpr int diagCov[5] = {0, 5, 9, 12, 14};
          float chi2 = 0.f;
          for (int p = 0; p < 5; ++p) {
            auto const variance = track_view[i].covariance()(diagCov[p]) + track_view[j].covariance()(diagCov[p]);
            if (variance <= 0.f)
              return;

            auto delta = track_view[i].state()(p) - track_view[j].state()(p);
            if (p == 0) {
              if (delta > M_PI)
                delta -= 2.f * M_PI;
              else if (delta < -M_PI)
                delta += 2.f * M_PI;
            }
            chi2 += delta * delta / variance;
          }

          if (chi2 > nSigma2)
            return;

          track_view[better(i, j) ? j : i].quality() = pixelTrack::Quality::dup;
#ifdef GPU_DEBUG
          printf("Kernel_trackParameterDuplicates: i: %u, j: %u, chi2: %f, nSigma2: %f, quality: %d\n",
                 i,
                 j,
                 chi2,
                 nSigma2,
                 uint32_t(track_view[better(i, j) ? j : i].quality()));
          printf(
              "Kernel_trackParameterDuplicates: i: %u, j: %u, pti: %f, ptj: %f, deltaPhi: %f, deltaEta: %f, "
              "maxDeltaR2: %f, maxRelativePtDifference: %f\n",
              i,
              j,
              pti,
              ptj,
              deltaPhi,
              etai - track_view[j].eta(),
              maxDeltaR2,
              maxRelativePtDifference);
#endif
        }
      }
    }
  };

  class Kernel_trackDuplicates {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ::reco::TrackSoAView track_view,
                                  ::reco::TrackHitSoAView trackHit_view,
                                  const bool doSameHitsDuplicates,
                                  const bool doParamDuplicates,
                                  const double matchFraction,
                                  const int minHitsForDuplicate,
                                  const float nSigma2,
                                  const float maxDeltaR2,
                                  const float maxRelativePtDifference,
                                  uint32_t* isDuplicate) const {
#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc))
        printf("Kernel_trackDuplicates: nTracks: %u\n", track_view.nTracks());
#endif
      for (uint32_t i : cms::alpakatools::uniform_elements_x(acc, uint32_t(track_view.nTracks()))) {
        for (uint32_t j : cms::alpakatools::uniform_elements_y(acc, i + 1, uint32_t(track_view.nTracks()))) {
          auto const better = [&](uint32_t a, uint32_t b) {
            if (track_view[a].nLayers() != track_view[b].nLayers())
              return track_view[a].nLayers() > track_view[b].nLayers();
            if (track_view[a].quality() != track_view[b].quality())
              return track_view[a].quality() > track_view[b].quality();
            if (track_view[a].chi2() != track_view[b].chi2())
              return track_view[a].chi2() < track_view[b].chi2();
            return a < b;
          };

          auto const sameHits = [&](uint32_t a, uint32_t b) {
            auto const nHitsA = uint32_t(::reco::nHits(track_view, a));
            if (nHitsA == 0 or nHitsA != uint32_t(::reco::nHits(track_view, b)))
              return false;

            auto const hitBeginA = a == 0 ? 0 : track_view[a - 1].hitOffsets();
            auto const hitBeginB = track_view[b - 1].hitOffsets();
            uint32_t matchedHits = 0;
            for (uint32_t k = 0; k < nHitsA; ++k)
              matchedHits += trackHit_view[hitBeginA + k].id() == trackHit_view[hitBeginB + k].id();

            return double(matchedHits) / double(nHitsA) > matchFraction and
                   matchedHits >= uint32_t(minHitsForDuplicate);
          };

          auto const parameterDuplicate = [&](uint32_t a, uint32_t b) {
            if (::reco::charge(track_view, a) != ::reco::charge(track_view, b))
              return false;

            auto deltaPhi = ::reco::phi(track_view, a) - ::reco::phi(track_view, b);
            if (deltaPhi > M_PI)
              deltaPhi -= 2.f * M_PI;
            else if (deltaPhi < -M_PI)
              deltaPhi += 2.f * M_PI;

            auto square = [](auto x) { return x * x; };
            if (square(deltaPhi) + square(track_view[a].eta() - track_view[b].eta()) > maxDeltaR2)
              return false;

            auto const minPt = alpaka::math::min(acc, track_view[a].pt(), track_view[b].pt());
            if (minPt <= 0.f or
                alpaka::math::abs(acc, track_view[a].pt() - track_view[b].pt()) > maxRelativePtDifference * minPt)
              return false;

            constexpr int diagCov[5] = {0, 5, 9, 12, 14};
            float chi2 = 0.f;
            for (int p = 0; p < 5; ++p) {
              auto const variance = track_view[a].covariance()(diagCov[p]) + track_view[b].covariance()(diagCov[p]);
              if (variance <= 0.f)
                return false;

              auto delta = track_view[a].state()(p) - track_view[b].state()(p);
              if (p == 0) {
                if (delta > M_PI)
                  delta -= 2.f * M_PI;
                else if (delta < -M_PI)
                  delta += 2.f * M_PI;
              }
              chi2 += delta * delta / variance;
            }
            return chi2 <= nSigma2;
          };

          bool duplicateByHits = false;
          bool duplicateByParameters = false;

          if (doSameHitsDuplicates)
            duplicateByHits = sameHits(i, j);

          if (doParamDuplicates)
            duplicateByParameters = parameterDuplicate(i, j);

          if (!(duplicateByHits or duplicateByParameters))
            continue;

          auto const loser = better(i, j) ? j : i;
          alpaka::atomicOr(acc, &isDuplicate[loser], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  class Kernel_applyTrackDuplicates {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAView track_view,
                                  const uint32_t* isDuplicate) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, track_view.nTracks())) {
        if (isDuplicate[i] != 0)
          track_view[i].quality() = pixelTrack::Quality::dup;
      }
    }
  };

  class Kernel_filterTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAView track_view,
                                  ::reco::TrackHitSoAView trackHit_view,
                                  const ::reco::TrackSoAConstView& inpTrack_view,
                                  const ::reco::TrackHitSoAConstView& inpTrackHit_view,
                                  const pixelTrack::Quality minQuality,
                                  const double matchFraction) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        uint32_t auxOutputTkIndex = 0;
        uint32_t auxOutputHitIndex = 0;

        for (uint32_t i = 0; i < uint32_t(inpTrack_view.metadata().size()); ++i) {
          if (inpTrack_view[i].quality() < minQuality)
            continue;

          bool hasDuplicate = false;
          for (uint32_t j : cms::alpakatools::uniform_elements(acc, inpTrack_view.metadata().size())) {
            if (j < i + 1)
              continue;
            if (inpTrack_view[j].quality() < minQuality)
              continue;

            if (::reco::nHits(inpTrack_view, i) == ::reco::nHits(inpTrack_view, j)) {
              uint32_t matchedHits = 0;
              for (uint32_t k = 0; k < uint32_t(::reco::nHits(inpTrack_view, i)); ++k) {
                uint32_t auxHitOffsetsId = 0;
                if (i > 0)
                  auxHitOffsetsId = inpTrack_view[i - 1].hitOffsets();
                if (inpTrackHit_view[auxHitOffsetsId + k].id() ==
                    inpTrackHit_view[inpTrack_view[j - 1].hitOffsets() + k].id())
                  ++matchedHits;
              }
              if (double(matchedHits) / double(::reco::nHits(inpTrack_view, i)) > matchFraction)
                hasDuplicate = true;
            }

            if (hasDuplicate)
              break;
          }
          alpaka::syncBlockThreads(acc);

          if (hasDuplicate)
            continue;

          track_view[auxOutputTkIndex].quality() = inpTrack_view[i].quality();
          track_view[auxOutputTkIndex].chi2() = inpTrack_view[i].chi2();
          track_view[auxOutputTkIndex].nLayers() = inpTrack_view[i].nLayers();
          track_view[auxOutputTkIndex].eta() = inpTrack_view[i].eta();
          track_view[auxOutputTkIndex].pt() = inpTrack_view[i].pt();
          for (uint32_t k = 0; k < 5; ++k)
            track_view[auxOutputTkIndex].state()[k] = inpTrack_view[i].state()[k];
          for (uint32_t k = 0; k < 15; ++k)
            track_view[auxOutputTkIndex].covariance()[k] = inpTrack_view[i].covariance()[k];
          if (auxOutputTkIndex != 0) {
            track_view[auxOutputTkIndex].hitOffsets() =
                track_view[auxOutputTkIndex - 1].hitOffsets() + ::reco::nHits(inpTrack_view, i);
          } else {
            track_view[auxOutputTkIndex].hitOffsets() = ::reco::nHits(inpTrack_view, i);
          }

          uint32_t auxHitOffsetsIdBegin = 0;
          if (i > 0)
            auxHitOffsetsIdBegin = inpTrack_view[i - 1].hitOffsets();

          uint32_t auxHitOffsetsIdEnd = inpTrack_view[i].hitOffsets();
          if (i > 0)
            auxHitOffsetsIdEnd = inpTrack_view[i].hitOffsets();

          for (uint32_t k = auxHitOffsetsIdBegin; k < auxHitOffsetsIdEnd; ++k) {
            trackHit_view[auxOutputHitIndex].id() = inpTrackHit_view[k].id();
            trackHit_view[auxOutputHitIndex].detId() = inpTrackHit_view[k].detId();
            ++auxOutputHitIndex;
          }

          ++auxOutputTkIndex;
        }
        alpaka::syncBlockThreads(acc);
        track_view.nTracks() = auxOutputTkIndex;
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::trackSoAMergerKernels

#endif  // RecoTracker_FinalTrackSelectors_plugins_alpaka_TrackSoAMergerKernelsImpl_h
