#ifndef RecoTracker_LSTCore_src_alpaka_PixelQuintupletAccessors_h
#define RecoTracker_LSTCore_src_alpaka_PixelQuintupletAccessors_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelQuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // A pT5's hits: the anchor and outer hits of its pLS's two pixel MDs, then its T5's hit slots (sentinels
  // beyond the T5's layers); identical to the per-pT5 copy addPixelQuintupletToMemory used to cache.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void getPixelQuintupletHitIndices(MiniDoubletsConst mds,
                                                                   SegmentsConst segments,
                                                                   QuintupletsConst quintuplets,
                                                                   PixelQuintupletsConst pixelQuintuplets,
                                                                   unsigned int pixelQuintupletIndex,
                                                                   unsigned int (&hitIndices)[Params_pT5::kHits]) {
    unsigned int const pixelIndex = pixelQuintuplets.pixelSegmentIndices()[pixelQuintupletIndex];
    unsigned int const pixelInnerMD = segments.mdIndices()[pixelIndex][0];
    unsigned int const pixelOuterMD = segments.mdIndices()[pixelIndex][1];
    hitIndices[0] = mds.anchorHitIndices()[pixelInnerMD];
    hitIndices[1] = mds.outerHitIndices()[pixelInnerMD];
    hitIndices[2] = mds.anchorHitIndices()[pixelOuterMD];
    hitIndices[3] = mds.outerHitIndices()[pixelOuterMD];
    auto const& t5Hits = quintuplets.hitIndices()[pixelQuintuplets.quintupletIndices()[pixelQuintupletIndex]];
    for (int i = 0; i < Params_T5::kHits; ++i)
      hitIndices[Params_pLS::kHits + i] = t5Hits[i];
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
