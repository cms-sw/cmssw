#ifndef RecoTracker_LSTCore_src_alpaka_Hit_h
#define RecoTracker_LSTCore_src_alpaka_Hit_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/alpakastdAlgorithm.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/HitsDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhiChange(TAcc const& acc, float x1, float y1, float x2, float y2) {
    return cms::alpakatools::deltaPhi(acc, x1, y1, x2 - x1, y2 - y1);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int packedHitIdx(unsigned int ih,
                                                           HitsBaseConst hitsBase,
                                                           HitsITConst hitsIT) {
    constexpr int kOTBit = 1 << 31;
    return hitOrigIdx(hitsBase, hitsIT, ih) | (hitsBase.detid()[ih] == kPixelModuleId ? 0 : kOTBit);
  }

  struct ModuleRangesKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  HitsRanges hitsRanges,
                                  int nLowerModules) const {
      for (int lowerIndex : cms::alpakatools::uniform_elements(acc, nLowerModules)) {
        uint16_t upperIndex = modules.partnerModuleIndices()[lowerIndex];
        if (hitsRanges.hitRanges()[lowerIndex][0] != -1 && hitsRanges.hitRanges()[upperIndex][0] != -1) {
          hitsRanges.hitRangesLower()[lowerIndex] = hitsRanges.hitRanges()[lowerIndex][0];
          hitsRanges.hitRangesUpper()[lowerIndex] = hitsRanges.hitRanges()[upperIndex][0];
          hitsRanges.hitRangesnLower()[lowerIndex] =
              hitsRanges.hitRanges()[lowerIndex][1] - hitsRanges.hitRanges()[lowerIndex][0] + 1;
          hitsRanges.hitRangesnUpper()[lowerIndex] =
              hitsRanges.hitRanges()[upperIndex][1] - hitsRanges.hitRanges()[upperIndex][0] + 1;
        }
      }
    }
  };

  struct HitLoopKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  unsigned int nModules,  // Number of modules
                                  ModulesConst modules,
                                  HitsBaseConst hitsBase,
                                  HitsExtended hitsExtended,
                                  HitsRanges hitsRanges) const  // Total number of hits in event
    {
      int nHits = hitsExtended.metadata().size();
      auto const nHitsOT = hitsBase.nHitsOT();
      ALPAKA_ASSERT_ACC(nHits == hitsBase.metadata().size());
      for (unsigned int ihit : cms::alpakatools::uniform_elements(acc, nHits)) {
        float ihit_x = hitsBase.xs()[ihit];
        float ihit_y = hitsBase.ys()[ihit];
        int iDetId = hitsBase.detid()[ihit];

        hitsExtended.rts()[ihit] = alpaka::math::sqrt(acc, ihit_x * ihit_x + ihit_y * ihit_y);
        hitsExtended.phis()[ihit] = cms::alpakatools::phi(acc, ihit_x, ihit_y);
        auto found_pointer =
            alpaka_std::lower_bound(modules.mapdetId().data(), modules.mapdetId().data() + nModules, iDetId);
        ALPAKA_ASSERT_ACC(found_pointer != modules.mapdetId().data() + nModules);
        int found_index = std::distance(modules.mapdetId().data(), found_pointer);
        uint16_t lastModuleIndex = modules.mapIdx()[found_index];

        hitsExtended.moduleIndices()[ihit] = lastModuleIndex;

        // hits above nHitsOT are from seed tracks: don't reindex the full OT hits (all below nHitsOT)
        if (ihit < nHitsOT || iDetId == kPixelModuleId) {
          // Need to set initial value if index hasn't been seen before.
          int old = alpaka::atomicCas(acc,
                                      &(hitsRanges.hitRanges()[lastModuleIndex][0]),
                                      -1,
                                      static_cast<int>(ihit),
                                      alpaka::hierarchy::Threads{});
          // For subsequent visits, stores the min value.
          if (old != -1)
            alpaka::atomicMin(
                acc, &hitsRanges.hitRanges()[lastModuleIndex][0], static_cast<int>(ihit), alpaka::hierarchy::Threads{});

          alpaka::atomicMax(
              acc, &hitsRanges.hitRanges()[lastModuleIndex][1], static_cast<int>(ihit), alpaka::hierarchy::Threads{});
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
