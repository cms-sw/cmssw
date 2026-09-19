#ifndef RecoTracker_LSTCore_src_alpaka_TripletAccessors_h
#define RecoTracker_LSTCore_src_alpaka_TripletAccessors_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // A triplet's six hit indices are the anchor and outer hits of its three
  // mini-doublets, reachable through its two segments. They used to be cached in
  // a per-triplet column; recovering them here instead costs one extra
  // indirection at the three places that consume them and saves 24 bytes on
  // every triplet slot. This is exactly the gather addTripletToMemory performed.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void getTripletMDIndices(SegmentsConst segments,
                                                          TripletsConst triplets,
                                                          unsigned int tripletIndex,
                                                          unsigned int (&mdIndices)[Params_T3::kLayers]) {
    unsigned int innerSegmentIndex = triplets.segmentIndices()[tripletIndex][0];
    unsigned int outerSegmentIndex = triplets.segmentIndices()[tripletIndex][1];
    mdIndices[0] = segments.mdIndices()[innerSegmentIndex][0];
    mdIndices[1] = segments.mdIndices()[innerSegmentIndex][1];
    mdIndices[2] = segments.mdIndices()[outerSegmentIndex][1];
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void getTripletHitIndices(MiniDoubletsConst mds,
                                                           SegmentsConst segments,
                                                           TripletsConst triplets,
                                                           unsigned int tripletIndex,
                                                           unsigned int (&hitIndices)[Params_T3::kHits]) {
    unsigned int mdIndices[Params_T3::kLayers];
    getTripletMDIndices(segments, triplets, tripletIndex, mdIndices);
    for (int i = 0; i < Params_T3::kLayers; ++i) {
      hitIndices[2 * i] = mds.anchorHitIndices()[mdIndices[i]];
      hitIndices[2 * i + 1] = mds.outerHitIndices()[mdIndices[i]];
    }
  }

  // The last mini-doublet of a triplet: the outer MD of its outer segment. This
  // is the tail that getTripletHitIndices reaches as mdIndices[2]; a consumer
  // needing only the triplet's final two hits can take this and skip the rest.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int getTripletLastMDIndex(SegmentsConst segments,
                                                                    TripletsConst triplets,
                                                                    unsigned int tripletIndex) {
    return segments.mdIndices()[triplets.segmentIndices()[tripletIndex][1]][1];
  }

  // The logical layer of a lower module: its layer number, offset by the six
  // barrel layers when the module is in an endcap (subdet 4). This is what
  // Triplets.logicalLayers used to cache, one entry per lower module index.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint8_t getLogicalLayer(ModulesConst modules, uint16_t lowerModuleIndex) {
    return modules.layers()[lowerModuleIndex] + (modules.subdets()[lowerModuleIndex] == 4) * 6;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
