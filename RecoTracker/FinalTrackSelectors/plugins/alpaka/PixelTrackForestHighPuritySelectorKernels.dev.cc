#include <alpaka/alpaka.hpp>
#include <algorithm>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackForestHighPuritySelectorKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // ------------------------------------------------------------------------------
  // The forest scorer exists in two kernels (TreeScoreKernel, TreeScoreWarpKernel) that differ only
  // in how the work is spread over the accelerator. The three pieces that define the result -- the
  // feature gather, the single-tree walk and the margin -> score map -- are shared here, so the
  // column order, the split convention and the sigmoid have a single definition.

  // Feature columns consumed by the forest == the full PixelTrackFeaturesSoA width: 31 fit/cov and
  // CA hit/stub columns, then 4 merged-collection provenance columns, then 7 pixel-cluster columns.
  // Keyed off kNPixelTrackFeatures so the scorer and the model loader's range check agree on the
  // width of f[].
  inline constexpr int kNForestFeatures = kNPixelTrackFeatures;

  // Tracks a CPU block gathers before making one pass over the whole forest. The tile costs
  // kForestCpuTrackTile * kNPixelTrackFeatures * 4 B of block stack (86 kB at 512), enough to
  // amortise a tree over the tile while staying far below the CMSSW thread stack.
  inline constexpr int kForestCpuTrackTile = 512;

  // Feature vector in PixelTrackFeaturesSoA column order == the trained ABI. Pure loads, no
  // arithmetic, so the gather cannot differ between backends.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void loadForestFeatures(const PixelTrackFeaturesSoA::ConstView& trackFeatures,
                                                         const Idx i,
                                                         float (&f)[kNForestFeatures]) {
    f[0] = trackFeatures[i].chi2();
    f[1] = trackFeatures[i].dzError();
    f[2] = trackFeatures[i].dxyError();
    f[3] = trackFeatures[i].eta();
    f[4] = trackFeatures[i].nHits();
    f[5] = trackFeatures[i].phi();
    f[6] = trackFeatures[i].phiError();
    f[7] = trackFeatures[i].pt();
    f[8] = trackFeatures[i].qOverPtError();
    f[9] = trackFeatures[i].dzBS();
    f[10] = trackFeatures[i].dxyBS();
    f[11] = trackFeatures[i].nLayers();
    f[12] = trackFeatures[i].cotThetaError();
    f[13] = trackFeatures[i].covCotThetaDz();
    f[14] = trackFeatures[i].covDxyQOverPt();
    f[15] = trackFeatures[i].covPhiDxy();
    f[16] = trackFeatures[i].covPhiQOverPt();
    f[17] = trackFeatures[i].caFitChi2();
    f[18] = trackFeatures[i].psFrac();
    f[19] = trackFeatures[i].r0();
    f[20] = trackFeatures[i].nPS();
    f[21] = trackFeatures[i].spanZ();
    f[22] = trackFeatures[i].nStubs();
    f[23] = trackFeatures[i].logChi2Stub();
    f[24] = trackFeatures[i].kErr();
    f[25] = trackFeatures[i].dcaEst();
    f[26] = trackFeatures[i].nBarrel();
    f[27] = trackFeatures[i].rzChi2();
    f[28] = trackFeatures[i].meanStubKappa();
    f[29] = trackFeatures[i].leverArm();
    f[30] = trackFeatures[i].rMax();
    // Cols 31-34: merged-collection provenance, indexed only by a merged-collection model.
    f[31] = trackFeatures[i].nAttached();
    f[32] = trackFeatures[i].nOTExtras();
    f[33] = trackFeatures[i].iterationId();
    f[34] = trackFeatures[i].ndof();
    // Cols 35-41: pixel-cluster charge/shape, indexed only by a 42-feature model.
    f[35] = trackFeatures[i].minCharge();
    f[36] = trackFeatures[i].meanCharge();
    f[37] = trackFeatures[i].minChargeNorm();
    f[38] = trackFeatures[i].maxSizeY();
    f[39] = trackFeatures[i].meanSizeY();
    f[40] = trackFeatures[i].maxSizeX();
    f[41] = trackFeatures[i].nLowCharge();
  }

  // Walk one tree from `node` (its root) down to a leaf and return the leaf value.
  // XGBoost convention: strict '<' on the fp32 threshold goes left; feat >= 0 -> split, -1 -> leaf.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float forestTreeLeaf(const int8_t* treeFeat,
                                                      const float* treeVal,
                                                      const int32_t* treeLeft,
                                                      const int32_t* treeRight,
                                                      int node,
                                                      const float (&f)[kNForestFeatures]) {
    while (treeFeat[node] >= 0)
      node = (f[treeFeat[node]] < treeVal[node]) ? treeLeft[node] : treeRight[node];
    return treeVal[node];
  }

  // margin -> score: the logistic map used by every scorer variant.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float forestScoreFromMargin(TAcc const& acc, const float margin) {
    return 1.f / (1.f + alpaka::math::exp(acc, -margin));
  }

  // ------------------------------------------------------------------------------

  // Compact gradient-boosted-tree scorer: a direct traversal of the forest from a per-device read-only buffer
  // (int8 feature index [-1 = leaf], float threshold / leaf value, int32 children, per-tree root offsets, base
  // margin); feature < threshold goes left. Tree-outer / track-inner over a tile of kForestCpuTrackTile tracks,
  // for the single-thread-per-block backends only; TreeScoreWarpKernel serves the others. The leaves are
  // accumulated in ascending tree order. Rows in [nValid, maxPreselectedTracks) are untouched.
  struct TreeScoreKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
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
                                  PixelTrackScoresSoA::View trackScores) const {
      static_assert(cms::alpakatools::requires_single_thread_per_block_v<TAcc>,
                    "TreeScoreKernel is the single-thread-per-block variant; the others use TreeScoreWarpKernel.");
      const auto nValid = alpaka::math::min(acc, *nPreselectedTracks, maxPreselectedTracks);
      // Tree-outer / track-inner, one pass over the forest per tile of tracks.
      // Per-tile scratch on the block's stack; one thread per block here, so it is block-private.
      float features[kForestCpuTrackTile][kNForestFeatures];
      float margins[kForestCpuTrackTile];
      // Elements per block == the tile the launcher asked for; the inner chunking below keeps the
      // kernel correct for any 1D work division.
      const Idx elementsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
      for (auto group : cms::alpakatools::uniform_groups(acc, nValid)) {
        const Idx groupBegin = group * elementsPerBlock;
        const Idx groupEnd = cms::alpakatools::idx_min(groupBegin + elementsPerBlock, static_cast<Idx>(nValid));
        for (Idx tileBegin = groupBegin; tileBegin < groupEnd; tileBegin += Idx(kForestCpuTrackTile)) {
          const int nTile = static_cast<int>(cms::alpakatools::idx_min(Idx(kForestCpuTrackTile), groupEnd - tileBegin));
          // Gather the tile's feature rows once; row-major, so a track's columns are adjacent.
          for (int k = 0; k < nTile; ++k) {
            loadForestFeatures(trackFeatures, tileBegin + Idx(k), features[k]);
            margins[k] = baseLogit;
          }
          // One pass over the forest for the whole tile; the inner walks are independent, so the
          // core overlaps several of their dependent load chains.
          for (int t = 0; t < nTrees; ++t) {
            const int root = treeRoots[t];
            for (int k = 0; k < nTile; ++k)
              margins[k] += forestTreeLeaf(treeFeat, treeVal, treeLeft, treeRight, root, features[k]);
          }
          for (int k = 0; k < nTile; ++k)
            trackScores[tileBegin + Idx(k)].score() = forestScoreFromMargin(acc, margins[k]);
        }
      }
    }
  };

  // Warp-per-track variant of TreeScoreKernel for the GPU backends: each of the warpSize lanes walks the trees
  // t = lane (mod warpSize) and the partial logits are combined by a fixed ascending butterfly all-reduce, so
  // the score is bit-stable per backend; it agrees with TreeScoreKernel to fp32 rounding (different
  // association). Work division: Acc2D, Y indexes tracks, X the warpSize lanes, so a warp is never split
  // across tracks and every shfl sees a converged warp.
  struct TreeScoreWarpKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
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
                                  PixelTrackScoresSoA::View trackScores) const {
      const auto nValid = alpaka::math::min(acc, *nPreselectedTracks, maxPreselectedTracks);
      const int32_t warpSize = alpaka::warp::getSize(acc);
      const int32_t laneId = static_cast<int32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1u]);
      for (auto i : cms::alpakatools::uniform_elements_y(acc, nValid)) {
        // All warpSize lanes read the same track i -> broadcast (single-address) global loads.
        float f[kNForestFeatures];
        loadForestFeatures(trackFeatures, i, f);
        // Lane 0 seeds baseLogit (added exactly once); every lane sums its own disjoint tree
        // subset.
        float partial = (laneId == 0) ? baseLogit : 0.f;
        for (int t = laneId; t < nTrees; t += warpSize)
          partial += forestTreeLeaf(treeFeat, treeVal, treeLeft, treeRight, treeRoots[t], f);
        // Fixed ascending-butterfly all-reduce (shfl from source lane laneId^off); every lane ends
        // with the full margin. A no-op on warpSize == 1.
        for (int32_t off = 1; off < warpSize; off <<= 1)
          partial += alpaka::warp::shfl(acc, partial, laneId ^ off);
        if (laneId == 0)
          trackScores[i].score() = forestScoreFromMargin(acc, partial);
      }
    }
  };

  // ------------------------------------------------------------------------------

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
                       PixelTrackScoresSoA::View trackScores) {
    if constexpr (cms::alpakatools::requires_single_thread_per_block_v<Acc1D>) {
      // CPU backends: warpSize == 1, so the warp path degenerates to one track per thread, each
      // track walking the whole multi-MB model on its own. TreeScoreKernel gathers kForestCpuTrackTile
      // tracks first and then walks each tree for all of them, so a tree is read once per tile
      // instead of once per track; measured 18.5 vs 30.4 ms/event on the serial arm.
      const auto blocks = cms::alpakatools::divide_up_by(maxPreselectedTracks, kForestCpuTrackTile);
      const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, kForestCpuTrackTile);
      alpaka::exec<Acc1D>(queue,
                          workDiv,
                          TreeScoreKernel{},
                          maxPreselectedTracks,
                          treeFeat,
                          treeVal,
                          treeLeft,
                          treeRight,
                          treeRoots,
                          nTrees,
                          baseLogit,
                          trackFeatures,
                          nPreselectedTracks,
                          trackScores);
    } else {
      // Warp-per-track: Y indexes tracks, X the warpSize lanes.
      const uint32_t warpSize = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
      const uint32_t tracksPerBlock = 4u;  // 4 tracks * warpSize lanes (=128 threads on CUDA)
      // gridDim.y is capped at 65535 by CUDA; uniform_elements_y strides any overflow.
      auto numBlocks = cms::alpakatools::divide_up_by(uint32_t(maxPreselectedTracks), tracksPerBlock);
      numBlocks = std::min<uint32_t>(std::max<uint32_t>(numBlocks, 1u), 65535u);
      const Vec2D blocks{numBlocks, 1u};
      const Vec2D threads{tracksPerBlock, warpSize};
      const auto workDiv = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);
      alpaka::exec<Acc2D>(queue,
                          workDiv,
                          TreeScoreWarpKernel{},
                          maxPreselectedTracks,
                          treeFeat,
                          treeVal,
                          treeLeft,
                          treeRight,
                          treeRoots,
                          nTrees,
                          baseLogit,
                          trackFeatures,
                          nPreselectedTracks,
                          trackScores);
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
