#ifndef RecoTracker_LSTCore_src_alpaka_Kernels_h
#define RecoTracker_LSTCore_src_alpaka_Kernels_h

#include <bit>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelQuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuadrupletsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmQuintupletFromMemory(Quintuplets quintuplets,
                                                             unsigned int quintupletIndex,
                                                             bool secondpass = false) {
    quintuplets.isDup()[quintupletIndex] |= 1 + secondpass;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelTripletFromMemory(PixelTriplets pixelTriplets,
                                                               unsigned int pixelTripletIndex) {
    pixelTriplets.isDup()[pixelTripletIndex] = true;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelQuintupletFromMemory(PixelQuintuplets pixelQuintuplets,
                                                                  unsigned int pixelQuintupletIndex) {
    pixelQuintuplets.isDup()[pixelQuintupletIndex] = true;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelSegmentFromMemory(PixelSegments pixelSegments,
                                                               unsigned int pixelSegmentArrayIndex,
                                                               bool secondpass = false) {
    pixelSegments.isDup()[pixelSegmentArrayIndex] |= 1 + secondpass;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmQuadrupletFromMemory(Quadruplets quadruplets,
                                                             unsigned int quadrupletIndex,
                                                             bool secondpass = false) {
    quadruplets.isDup()[quadrupletIndex] |= 1 + secondpass;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitsT5(unsigned int ix, unsigned int jx, QuintupletsConst quintuplets) {
    unsigned int hits1[Params_T5::kHits];
    unsigned int hits2[Params_T5::kHits];

    for (int i = 0; i < Params_T5::kHits; i++) {
      hits1[i] = quintuplets.hitIndices()[ix][i];
      hits2[i] = quintuplets.hitIndices()[jx][i];
    }

    int nMatched = 0;
    for (int i = 0; i < Params_T5::kHits; i++) {
      // Skip sentinel values from extended slots
      if (hits1[i] == lst::kTCEmptyHitIdx)
        continue;
      bool matched = false;
      for (int j = 0; j < Params_T5::kHits; j++) {
        if (hits2[j] == lst::kTCEmptyHitIdx)
          continue;
        if (hits1[i] == hits2[j]) {
          matched = true;
          break;
        }
      }
      if (matched) {
        nMatched++;
      }
    }
    return nMatched;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitspT5(unsigned int ix,
                                                  unsigned int jx,
                                                  PixelQuintupletsConst pixelQuintuplets) {
    unsigned int hits1[Params_pT5::kHits];
    unsigned int hits2[Params_pT5::kHits];

    for (int i = 0; i < Params_pT5::kHits; i++) {
      hits1[i] = pixelQuintuplets.hitIndices()[ix][i];
      hits2[i] = pixelQuintuplets.hitIndices()[jx][i];
    }

    int nMatched = 0;
    for (int i = 0; i < Params_pT5::kHits; i++) {
      // Skip sentinel values from extended slots
      if (hits1[i] == lst::kTCEmptyHitIdx)
        continue;
      bool matched = false;
      for (int j = 0; j < Params_pT5::kHits; j++) {
        if (hits2[j] == lst::kTCEmptyHitIdx)
          continue;
        if (hits1[i] == hits2[j]) {
          matched = true;
          break;
        }
      }
      if (matched) {
        nMatched++;
      }
    }
    return nMatched;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void checkHitspT3(unsigned int ix,
                                                   unsigned int jx,
                                                   PixelTripletsConst pixelTriplets,
                                                   int* matched) {
    int phits1[Params_pLS::kHits];
    int phits2[Params_pLS::kHits];

    for (int i = 0; i < Params_pLS::kHits; i++) {
      phits1[i] = pixelTriplets.hitIndices()[ix][i];
      phits2[i] = pixelTriplets.hitIndices()[jx][i];
    }

    int npMatched = 0;
    for (int i = 0; i < Params_pLS::kHits; i++) {
      bool pmatched = false;
      for (int j = 0; j < Params_pLS::kHits; j++) {
        if (phits1[i] == phits2[j]) {
          pmatched = true;
          break;
        }
      }
      if (pmatched) {
        npMatched++;
      }
    }

    int hits1[Params_T3::kHits];
    int hits2[Params_T3::kHits];

    for (int i = 0; i < Params_T3::kHits; i++) {
      hits1[i] = pixelTriplets.hitIndices()[ix][i + 4];  // Omitting the pLS hits
      hits2[i] = pixelTriplets.hitIndices()[jx][i + 4];  // Omitting the pLS hits
    }

    int nMatched = 0;
    for (int i = 0; i < Params_T3::kHits; i++) {
      bool tmatched = false;
      for (int j = 0; j < Params_T3::kHits; j++) {
        if (hits1[i] == hits2[j]) {
          tmatched = true;
          break;
        }
      }
      if (tmatched) {
        nMatched++;
      }
    }

    matched[0] = npMatched;
    matched[1] = nMatched;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitsT4(unsigned int ix, unsigned int jx, QuadrupletsConst quadruplets) {
    unsigned int hits1[Params_T4::kHits];
    unsigned int hits2[Params_T4::kHits];

    for (int i = 0; i < Params_T4::kHits; i++) {
      hits1[i] = quadruplets.hitIndices()[ix][i];
      hits2[i] = quadruplets.hitIndices()[jx][i];
    }

    int nMatched = 0;
    for (int i = 0; i < Params_T4::kHits; i++) {
      bool matched = false;
      for (int j = 0; j < Params_T4::kHits; j++) {
        if (hits1[i] == hits2[j]) {
          matched = true;
          break;
        }
      }
      if (matched) {
        nMatched++;
      }
    }
    return nMatched;
  };

  struct RemoveDupQuintupletsAfterBuild {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  ObjectRangesConst ranges) const {
      for (unsigned int lowmod : cms::alpakatools::uniform_elements_z(acc, modules.nLowerModules())) {
        unsigned int nQuintuplets_lowmod = quintupletsOccupancy.nQuintuplets()[lowmod];
        int quintupletModuleIndices_lowmod = ranges.quintupletModuleIndices()[lowmod];

        for (unsigned int ix1 : cms::alpakatools::uniform_elements_y(acc, nQuintuplets_lowmod)) {
          unsigned int ix = quintupletModuleIndices_lowmod + ix1;
          if (quintuplets.isDup()[ix])
            continue;
          float eta1 = __H2F(quintuplets.eta()[ix]);
          float phi1 = __H2F(quintuplets.phi()[ix]);
          float dnnScore1 = quintuplets.dnnScore()[ix];

          for (unsigned int jx1 : cms::alpakatools::uniform_elements_x(acc, ix1 + 1, nQuintuplets_lowmod)) {
            unsigned int jx = quintupletModuleIndices_lowmod + jx1;
            if (quintuplets.isDup()[jx])
              continue;

            float eta2 = __H2F(quintuplets.eta()[jx]);
            float phi2 = __H2F(quintuplets.phi()[jx]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

            if (dEta > 0.1f)
              continue;

            if (alpaka::math::abs(acc, dPhi) > 0.1f)
              continue;

            int nMatched = checkHitsT5(ix, jx, quintuplets);
            // Proportional sharing: at least 60% of the shorter track's hits.
            unsigned int nLayersIx = quintuplets.nLayers()[ix];
            unsigned int nLayersJx = quintuplets.nLayers()[jx];
            unsigned int nHitsIx = 2 * nLayersIx;
            unsigned int nHitsJx = 2 * nLayersJx;
            int minNHitsForDup = static_cast<int>(0.6f * (nHitsIx < nHitsJx ? nHitsIx : nHitsJx));
            if (nMatched >= minNHitsForDup) {
              // Tiebreak: longer track wins; otherwise rphisum at high pT, DNN score at low pT.
              if (nLayersIx > nLayersJx) {
                rmQuintupletFromMemory(quintuplets, jx);
              } else if (nLayersJx > nLayersIx) {
                rmQuintupletFromMemory(quintuplets, ix);
              } else {
                float ptIx = __H2F(quintuplets.innerRadius()[ix]) * lst::k2Rinv1GeVf * 2;
                float ptJx = __H2F(quintuplets.innerRadius()[jx]) * lst::k2Rinv1GeVf * 2;
                if (ptIx > 5.0f || ptJx > 5.0f) {
                  float rphisum1 = __H2F(quintuplets.score_rphisum()[ix]);
                  float rphisum2 = __H2F(quintuplets.score_rphisum()[jx]);
                  if (rphisum1 >= rphisum2)
                    rmQuintupletFromMemory(quintuplets, ix);
                  else
                    rmQuintupletFromMemory(quintuplets, jx);
                } else {
                  float dnnScore2 = quintuplets.dnnScore()[jx];
                  if (dnnScore1 <= dnnScore2)
                    rmQuintupletFromMemory(quintuplets, ix);
                  else
                    rmQuintupletFromMemory(quintuplets, jx);
                }
              }
            }
          }
        }
      }
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void tryExtendT5(
      TAcc const& acc, Quintuplets quintuplets, unsigned int winnerIdx, unsigned int loserIdx, int loserSlot) {
    if (loserSlot < 0)
      return;

    unsigned int newSlot = alpaka::atomicAdd(acc, &quintuplets.nLayers()[winnerIdx], 1u, alpaka::hierarchy::Threads{});

    if (newSlot >= Params_T5::kLayers) {
      alpaka::atomicSub(acc, &quintuplets.nLayers()[winnerIdx], 1u, alpaka::hierarchy::Threads{});
      return;
    }

    quintuplets.logicalLayers()[winnerIdx][newSlot] = quintuplets.logicalLayers()[loserIdx][loserSlot];
    quintuplets.lowerModuleIndices()[winnerIdx][newSlot] = quintuplets.lowerModuleIndices()[loserIdx][loserSlot];
    quintuplets.hitIndices()[winnerIdx][2 * newSlot] = quintuplets.hitIndices()[loserIdx][2 * loserSlot];
    quintuplets.hitIndices()[winnerIdx][2 * newSlot + 1] = quintuplets.hitIndices()[loserIdx][2 * loserSlot + 1];
  }

  struct ExtendT5FromDupT5 {
    // Packed [score:32 | T5 index:28 | layer slot:4] for atomic best-per-OT-layer tracking.
    static constexpr int kPackedScoreShift = 32;
    static constexpr int kPackedIndexShift = 4;
    static constexpr unsigned int kPackedIndexMask = 0xFFFFFFF;
    static constexpr unsigned int kPackedSlotMask = 0xF;
    static constexpr int kT5DuplicateMinSharedHits = 8;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy) const {
      // Best candidate per OT logical layer (1..11), packed score|index|slot.
      uint64_t* sharedBestPacked = alpaka::declareSharedVar<uint64_t[lst::kLogicalOTLayers], __COUNTER__>(acc);

      // One block per T5 in 1D; block index = ref T5 index.
      const unsigned int refT5Index = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];

      // Skip empty/unallocated T5 slots.
      if (quintuplets.nLayers()[refT5Index] == 0)
        return;

      // Initialize shared memory once per block.
      if (cms::alpakatools::once_per_block(acc)) {
        for (int logicalLayerBin = 0; logicalLayerBin < lst::kLogicalOTLayers; ++logicalLayerBin) {
          sharedBestPacked[logicalLayerBin] = 0;
        }
      }
      alpaka::syncBlockThreads(acc);

      const float baseEta = __H2F(quintuplets.eta()[refT5Index]);
      const float basePhi = __H2F(quintuplets.phi()[refT5Index]);
      const uint8_t baseStartLogicalLayer = quintuplets.logicalLayers()[refT5Index][0];

      // Hoist ref data once: hit indices and embedding read every candidate iteration otherwise.
      float refEmbed[Params_T5::kEmbed];
      CMS_UNROLL_LOOP
      for (unsigned int e = 0; e < Params_T5::kEmbed; ++e)
        refEmbed[e] = quintuplets.t5Embed()[refT5Index][e];

      constexpr unsigned int kRefHits = 2 * Params_T5::kBaseLayers;
      unsigned int refHits[kRefHits];
      CMS_UNROLL_LOOP
      for (unsigned int h = 0; h < kRefHits; ++h)
        refHits[h] = quintuplets.hitIndices()[refT5Index][h];

      const int threadIndexFlat = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      const int blockDimFlat = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

      // Ref-starts-at-layer-1 special case: only ref's slot-1 module can host a valid candidate.
      const bool restrictToRefSlot1 = (baseStartLogicalLayer == 1);
      const uint16_t nEligibleT5Modules = ranges.nEligibleT5Modules();
      const int loopCount = restrictToRefSlot1 ? 1 : static_cast<int>(nEligibleT5Modules);

      for (int idx = threadIndexFlat; idx < loopCount; idx += blockDimFlat) {
        const uint16_t lowerModuleIndex = restrictToRefSlot1 ? quintuplets.lowerModuleIndices()[refT5Index][1]
                                                             : ranges.indicesOfEligibleT5Modules()[idx];

        if (!restrictToRefSlot1) {
          // Skip same-starting-layer modules (logical = physical + 6 for endcap, see Triplet.h).
          const short modSubdet = modules.subdets()[lowerModuleIndex];
          const int moduleLogicalLayer =
              static_cast<int>(modules.layers()[lowerModuleIndex]) + (modSubdet == Endcap ? 6 : 0);
          if (moduleLogicalLayer == static_cast<int>(baseStartLogicalLayer))
            continue;

          // Module-level eta/phi pre-cut; margin covers per-T5 window plus T5-vs-module spread.
          if (alpaka::math::abs(acc, baseEta - modules.eta()[lowerModuleIndex]) > 0.3f)
            continue;
          if (alpaka::math::abs(acc, cms::alpakatools::deltaPhi(acc, basePhi, modules.phi()[lowerModuleIndex])) > 0.5f)
            continue;
        }

        const int firstQuintupletInModule = ranges.quintupletModuleIndices()[lowerModuleIndex];
        if (firstQuintupletInModule == -1)
          continue;

        const unsigned int nQuintupletsInModule = quintupletsOccupancy.nQuintuplets()[lowerModuleIndex];

        for (unsigned int quintupletOffset = 0; quintupletOffset < nQuintupletsInModule; ++quintupletOffset) {
          const unsigned int testT5Index = firstQuintupletInModule + quintupletOffset;
          if (testT5Index == refT5Index)
            continue;

          // Per-T5 eta/phi window.
          const float candidateEta = __H2F(quintuplets.eta()[testT5Index]);
          if (alpaka::math::abs(acc, baseEta - candidateEta) > 0.1f)
            continue;

          const float candidatePhi = __H2F(quintuplets.phi()[testT5Index]);
          if (alpaka::math::abs(acc, cms::alpakatools::deltaPhi(acc, basePhi, candidatePhi)) > 0.1f)
            continue;

          // Embedding distance against hoisted refEmbed.
          float embedDistance2 = 0.f;
          CMS_UNROLL_LOOP
          for (unsigned int embedIndex = 0; embedIndex < Params_T5::kEmbed; ++embedIndex) {
            const float diff = refEmbed[embedIndex] - quintuplets.t5Embed()[testT5Index][embedIndex];
            embedDistance2 += diff * diff;
          }
          if (embedDistance2 > 1.0f)
            continue;

          // Hit matching against hoisted ref hits; record the candidate slot with no shared hit.
          int sharedHitCount = 0;
          int unmatchedLayerSlot = -1;
          CMS_UNROLL_LOOP
          for (unsigned int layerIndex = 0; layerIndex < Params_T5::kBaseLayers; ++layerIndex) {
            const unsigned int candidateHit0 = quintuplets.hitIndices()[testT5Index][2 * layerIndex + 0];
            const unsigned int candidateHit1 = quintuplets.hitIndices()[testT5Index][2 * layerIndex + 1];

            bool hit0InBase = false;
            bool hit1InBase = false;
            CMS_UNROLL_LOOP
            for (unsigned int baseHitIndex = 0; baseHitIndex < kRefHits; ++baseHitIndex) {
              const unsigned int baseHit = refHits[baseHitIndex];
              hit0InBase = hit0InBase || (candidateHit0 == baseHit);
              hit1InBase = hit1InBase || (candidateHit1 == baseHit);
            }

            sharedHitCount += int(hit0InBase) + int(hit1InBase);
            if (!hit0InBase && !hit1InBase)
              unmatchedLayerSlot = layerIndex;
          }

          if (sharedHitCount < kT5DuplicateMinSharedHits)
            continue;
          if (unmatchedLayerSlot < 0)
            continue;

          // Score = DNN output; layer bin = candidate's unmatched OT layer (1..11) - 1.
          const float candidateScore = quintuplets.dnnScore()[testT5Index];
          const uint8_t newLogicalLayer = quintuplets.logicalLayers()[testT5Index][unmatchedLayerSlot];
          const int logicalLayerBin = static_cast<int>(newLogicalLayer) - 1;

          uint64_t scoreBits = std::bit_cast<uint32_t>(candidateScore);
          uint64_t newPacked = (scoreBits << kPackedScoreShift) |
                               (static_cast<uint64_t>(testT5Index & kPackedIndexMask) << kPackedIndexShift) |
                               (unmatchedLayerSlot & kPackedSlotMask);

          // Atomic CAS into shared best-per-layer slot, retry until we win or are beaten.
          uint64_t oldPacked = sharedBestPacked[logicalLayerBin];
          while (true) {
            const float oldScore = std::bit_cast<float>(static_cast<uint32_t>(oldPacked >> kPackedScoreShift));
            if (candidateScore <= oldScore)
              break;

            uint64_t assumedOld = alpaka::atomicCas(
                acc, &sharedBestPacked[logicalLayerBin], oldPacked, newPacked, alpaka::hierarchy::Threads{});

            if (assumedOld == oldPacked) {
              break;
            } else {
              oldPacked = assumedOld;
            }
          }
        }
      }

      alpaka::syncBlockThreads(acc);

      // One thread per block applies the per-layer winners.
      if (cms::alpakatools::once_per_block(acc)) {
        CMS_UNROLL_LOOP
        for (int logicalLayerBin = 0; logicalLayerBin < lst::kLogicalOTLayers; ++logicalLayerBin) {
          uint64_t bestPacked = sharedBestPacked[logicalLayerBin];
          if ((bestPacked >> kPackedScoreShift) == 0)
            continue;

          const int bestT5Index = static_cast<int>((bestPacked >> kPackedIndexShift) & kPackedIndexMask);
          const int bestT5LayerSlot = static_cast<int>(bestPacked & kPackedSlotMask);

          tryExtendT5(acc, quintuplets, refT5Index, bestT5Index, bestT5LayerSlot);
        }
      }
    }
  };

  struct RemoveDupQuintupletsBeforeTC {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  ObjectRangesConst ranges) const {
      for (unsigned int lowmodIdx1 : cms::alpakatools::uniform_elements_y(acc, ranges.nEligibleT5Modules())) {
        uint16_t lowmod1 = ranges.indicesOfEligibleT5Modules()[lowmodIdx1];
        unsigned int nQuintuplets_lowmod1 = quintupletsOccupancy.nQuintuplets()[lowmod1];
        if (nQuintuplets_lowmod1 == 0)
          continue;

        unsigned int quintupletModuleIndices_lowmod1 = ranges.quintupletModuleIndices()[lowmod1];

        for (unsigned int lowmodIdx2 :
             cms::alpakatools::uniform_elements_x(acc, lowmodIdx1, ranges.nEligibleT5Modules())) {
          uint16_t lowmod2 = ranges.indicesOfEligibleT5Modules()[lowmodIdx2];
          unsigned int nQuintuplets_lowmod2 = quintupletsOccupancy.nQuintuplets()[lowmod2];
          if (nQuintuplets_lowmod2 == 0)
            continue;

          unsigned int quintupletModuleIndices_lowmod2 = ranges.quintupletModuleIndices()[lowmod2];

          for (unsigned int ix1 = 0; ix1 < nQuintuplets_lowmod1; ix1 += 1) {
            unsigned int ix = quintupletModuleIndices_lowmod1 + ix1;
            if (quintuplets.isDup()[ix] & 1)
              continue;

            const bool isPT5_ix = quintuplets.partOfPT5()[ix];
            const float eta1 = __H2F(quintuplets.eta()[ix]);
            const float phi1 = __H2F(quintuplets.phi()[ix]);
            const float dnnScore1 = quintuplets.dnnScore()[ix];

            for (unsigned int jx1 = 0; jx1 < nQuintuplets_lowmod2; jx1++) {
              unsigned int jx = quintupletModuleIndices_lowmod2 + jx1;
              if (ix == jx)
                continue;

              if (quintuplets.isDup()[jx] & 1)
                continue;

              const bool isPT5_jx = quintuplets.partOfPT5()[jx];

              if (isPT5_ix && isPT5_jx)
                continue;

              const float eta2 = __H2F(quintuplets.eta()[jx]);
              const float dEta = alpaka::math::abs(acc, eta1 - eta2);
              if (dEta > 0.1f)
                continue;

              const float phi2 = __H2F(quintuplets.phi()[jx]);
              const float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);
              if (alpaka::math::abs(acc, dPhi) > 0.1f)
                continue;

              const float dR2 = dEta * dEta + dPhi * dPhi;
              const int nMatched = checkHitsT5(ix, jx, quintuplets);
              constexpr int minNHitsForDup_T5 = 5;

              float d2 = 0.f;
              CMS_UNROLL_LOOP
              for (unsigned int k = 0; k < Params_T5::kEmbed; ++k) {
                float diff = quintuplets.t5Embed()[ix][k] - quintuplets.t5Embed()[jx][k];
                d2 += diff * diff;
              }

              // 99th percentile of true-dup d2 distribution measured on 100 PU200 events.
              constexpr float d2Thresh = 0.25f;
              if (((dR2 < 0.001f || nMatched >= minNHitsForDup_T5) && d2 < d2Thresh) || (dR2 < 0.02f && d2 < 0.1f)) {
                float ptIx = __H2F(quintuplets.innerRadius()[ix]) * lst::k2Rinv1GeVf * 2;
                float ptJx = __H2F(quintuplets.innerRadius()[jx]) * lst::k2Rinv1GeVf * 2;
                bool highPt = (ptIx > 5.0f || ptJx > 5.0f);
                bool ixLoses;
                if (isPT5_jx) {
                  ixLoses = true;
                } else if (isPT5_ix) {
                  ixLoses = false;
                } else if (highPt) {
                  float rphisum1 = __H2F(quintuplets.score_rphisum()[ix]);
                  float rphisum2 = __H2F(quintuplets.score_rphisum()[jx]);
                  ixLoses = (rphisum1 > rphisum2) || (rphisum1 == rphisum2 && ix < jx);
                } else {
                  float dnnScore2 = quintuplets.dnnScore()[jx];
                  ixLoses = (dnnScore1 < dnnScore2) || (dnnScore1 == dnnScore2 && ix < jx);
                }
                if (ixLoses)
                  rmQuintupletFromMemory(quintuplets, ix, true);
                else
                  rmQuintupletFromMemory(quintuplets, jx, true);
              }
            }
          }
        }
      }
    }
  };

  struct RemoveDupQuadrupletsAfterBuild {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  Quadruplets quadruplets,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
                                  ObjectRangesConst ranges) const {
      for (auto lowmod : cms::alpakatools::uniform_elements_z(acc, modules.nLowerModules())) {
        unsigned int nQuadruplets_lowmod = quadrupletsOccupancy.nQuadruplets()[lowmod];
        int quadrupletModuleIndices_lowmod = ranges.quadrupletModuleIndices()[lowmod];

        for (unsigned int ix1 : cms::alpakatools::uniform_elements_y(acc, nQuadruplets_lowmod)) {
          unsigned int ix = quadrupletModuleIndices_lowmod + ix1;
          const float eta1 = __H2F(quadruplets.eta()[ix]);
          const float phi1 = __H2F(quadruplets.phi()[ix]);
          const float score1 = quadruplets.displacedScore()[ix] - quadruplets.fakeScore()[ix];

          for (unsigned int jx1 : cms::alpakatools::uniform_elements_x(acc, ix1 + 1, nQuadruplets_lowmod)) {
            unsigned int jx = quadrupletModuleIndices_lowmod + jx1;

            const float eta2 = __H2F(quadruplets.eta()[jx]);
            const float phi2 = __H2F(quadruplets.phi()[jx]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

            if (dEta > 0.1f)
              continue;

            if (alpaka::math::abs(acc, dPhi) > 0.1f)
              continue;

            const float score2 = quadruplets.displacedScore()[jx] - quadruplets.fakeScore()[jx];

            int nMatched = checkHitsT4(ix, jx, quadruplets);
            const int minNHitsForDup_T4 = 5;
            if (nMatched >= minNHitsForDup_T4) {
              if (score1 >= score2) {
                rmQuadrupletFromMemory(quadruplets, jx);
              } else {
                rmQuadrupletFromMemory(quadruplets, ix);
              }
            }
          }
        }
      }
    }
  };

  struct RemoveDupQuadrupletsBeforeTC {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  Quadruplets quadruplets,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
                                  ObjectRangesConst ranges) const {
      for (unsigned int lowmodIdx1 : cms::alpakatools::uniform_elements_y(acc, ranges.nEligibleT4Modules())) {
        uint16_t lowmod1 = ranges.indicesOfEligibleT4Modules()[lowmodIdx1];
        unsigned int nQuadruplets_lowmod1 = quadrupletsOccupancy.nQuadruplets()[lowmod1];
        if (nQuadruplets_lowmod1 == 0)
          continue;

        unsigned int quadrupletModuleIndices_lowmod1 = ranges.quadrupletModuleIndices()[lowmod1];

        for (unsigned int lowmodIdx2 :
             cms::alpakatools::uniform_elements_x(acc, lowmodIdx1, ranges.nEligibleT4Modules())) {
          uint16_t lowmod2 = ranges.indicesOfEligibleT4Modules()[lowmodIdx2];
          unsigned int nQuadruplets_lowmod2 = quadrupletsOccupancy.nQuadruplets()[lowmod2];
          if (nQuadruplets_lowmod2 == 0)
            continue;

          unsigned int quadrupletModuleIndices_lowmod2 = ranges.quadrupletModuleIndices()[lowmod2];

          for (unsigned int ix1 = 0; ix1 < nQuadruplets_lowmod1; ix1 += 1) {
            unsigned int ix = quadrupletModuleIndices_lowmod1 + ix1;
            if ((quadruplets.isDup()[ix] & 1))
              continue;

            const float eta1 = __H2F(quadruplets.eta()[ix]);
            const float phi1 = __H2F(quadruplets.phi()[ix]);
            const float score1 = quadruplets.displacedScore()[ix] - quadruplets.fakeScore()[ix];

            for (unsigned int jx1 = 0; jx1 < nQuadruplets_lowmod2; jx1++) {
              unsigned int jx = quadrupletModuleIndices_lowmod2 + jx1;
              if (ix == jx)
                continue;

              if ((quadruplets.isDup()[jx] & 1))
                continue;

              const float eta2 = __H2F(quadruplets.eta()[jx]);
              const float phi2 = __H2F(quadruplets.phi()[jx]);
              float dEta = alpaka::math::abs(acc, eta1 - eta2);
              float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

              if (dEta > 0.1f)
                continue;

              if (alpaka::math::abs(acc, dPhi) > 0.1f)
                continue;

              const float score2 = quadruplets.displacedScore()[jx] - quadruplets.fakeScore()[jx];

              float dR2 = dEta * dEta + dPhi * dPhi;
              int nMatched = checkHitsT4(ix, jx, quadruplets);
              const int minNHitsForDup_T4 = 4;
              if (dR2 < 0.001f || nMatched >= minNHitsForDup_T4) {
                if (score1 > score2) {
                  rmQuadrupletFromMemory(quadruplets, jx, true);
                } else if (score1 < score2) {
                  rmQuadrupletFromMemory(quadruplets, ix, true);
                } else {
                  rmQuadrupletFromMemory(quadruplets, (ix < jx ? ix : jx), true);
                }
              }
            }
          }
        }
      }
    }
  };

  struct RemoveDupPixelTripletsFromMap {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc, PixelTriplets pixelTriplets) const {
      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, pixelTriplets.nPixelTriplets())) {
        for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, pixelTriplets.nPixelTriplets())) {
          if (ix == jx)
            continue;

          int nMatched[2];
          checkHitspT3(ix, jx, pixelTriplets, nMatched);
          const int minNHitsForDup_pT3 = 5;
          if ((nMatched[0] + nMatched[1]) >= minNHitsForDup_pT3) {
            // Check the layers
            if (pixelTriplets.logicalLayers()[jx][2] < pixelTriplets.logicalLayers()[ix][2]) {
              rmPixelTripletFromMemory(pixelTriplets, ix);
              break;
            } else if (pixelTriplets.logicalLayers()[ix][2] == pixelTriplets.logicalLayers()[jx][2] &&
                       __H2F(pixelTriplets.score()[ix]) > __H2F(pixelTriplets.score()[jx])) {
              rmPixelTripletFromMemory(pixelTriplets, ix);
              break;
            } else if (pixelTriplets.logicalLayers()[ix][2] == pixelTriplets.logicalLayers()[jx][2] &&
                       (__H2F(pixelTriplets.score()[ix]) == __H2F(pixelTriplets.score()[jx])) && (ix < jx)) {
              rmPixelTripletFromMemory(pixelTriplets, ix);
              break;
            }
          }
        }
      }
    }
  };

  struct RemoveDupPixelQuintupletsFromMap {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc, PixelQuintuplets pixelQuintuplets) const {
      unsigned int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, nPixelQuintuplets)) {
        float eta1 = __H2F(pixelQuintuplets.eta()[ix]);
        float phi1 = __H2F(pixelQuintuplets.phi()[ix]);
        float score1 = __H2F(pixelQuintuplets.score()[ix]);
        for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, nPixelQuintuplets)) {
          if (ix == jx)
            continue;

          float eta2 = __H2F(pixelQuintuplets.eta()[jx]);
          if (alpaka::math::abs(acc, eta1 - eta2) > 0.2f)
            continue;

          float phi2 = __H2F(pixelQuintuplets.phi()[jx]);
          if (alpaka::math::abs(acc, cms::alpakatools::deltaPhi(acc, phi1, phi2)) > 0.2f)
            continue;

          int nMatched = checkHitspT5(ix, jx, pixelQuintuplets);
          float score2 = __H2F(pixelQuintuplets.score()[jx]);
          const int minNHitsForDup_pT5 = 7;
          if (nMatched >= minNHitsForDup_pT5) {
            if (score1 > score2 or ((score1 == score2) and (ix > jx))) {
              rmPixelQuintupletFromMemory(pixelQuintuplets, ix);
              break;
            }
          }
        }
      }
    }
  };

  struct CheckHitspLS {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegments pixelSegments,
                                  bool secondpass) const {
      int pixelModuleIndex = modules.nLowerModules();
      unsigned int nPixelSegments = segmentsOccupancy.nSegments()[pixelModuleIndex];

      if (nPixelSegments > n_max_pixel_segments_per_module)
        nPixelSegments = n_max_pixel_segments_per_module;

      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, nPixelSegments)) {
        if (secondpass && (!pixelSeeds.isQuad()[ix] || (pixelSegments.isDup()[ix] & 1)))
          continue;

        auto const& phits1 = pixelSegments.pLSHitsIdxs()[ix];
        float eta_pix1 = pixelSeeds.eta()[ix];
        float phi_pix1 = pixelSeeds.phi()[ix];

        for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, ix + 1, nPixelSegments)) {
          float eta_pix2 = pixelSeeds.eta()[jx];
          float phi_pix2 = pixelSeeds.phi()[jx];

          if (alpaka::math::abs(acc, eta_pix2 - eta_pix1) > 0.1f)
            continue;

          if (secondpass && (!pixelSeeds.isQuad()[jx] || (pixelSegments.isDup()[jx] & 1)))
            continue;

          int8_t quad_diff = pixelSeeds.isQuad()[ix] - pixelSeeds.isQuad()[jx];
          float score_diff = pixelSegments.score()[ix] - pixelSegments.score()[jx];
          // Always keep quads over trips. If they are the same, we want the object with better score
          int idxToRemove;
          if (quad_diff > 0)
            idxToRemove = jx;
          else if (quad_diff < 0)
            idxToRemove = ix;
          else if (score_diff < 0)
            idxToRemove = jx;
          else if (score_diff > 0)
            idxToRemove = ix;
          else
            idxToRemove = ix;

          auto const& phits2 = pixelSegments.pLSHitsIdxs()[jx];

          int npMatched = 0;
          for (int i = 0; i < Params_pLS::kHits; i++) {
            bool pmatched = false;
            for (int j = 0; j < Params_pLS::kHits; j++) {
              if (phits1[i] == phits2[j]) {
                pmatched = true;
                break;
              }
            }
            if (pmatched) {
              npMatched++;
              // Only one hit is enough
              if (secondpass)
                break;
            }
          }
          const int minNHitsForDup_pLS = 3;
          if (npMatched >= minNHitsForDup_pLS) {
            rmPixelSegmentFromMemory(pixelSegments, idxToRemove, secondpass);
          }
          if (secondpass) {
            float dEta = alpaka::math::abs(acc, eta_pix1 - eta_pix2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi_pix1, phi_pix2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if ((npMatched >= 1) || (dR2 < 1e-5f)) {
              rmPixelSegmentFromMemory(pixelSegments, idxToRemove, secondpass);
            }
          }
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
