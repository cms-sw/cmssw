#ifndef RecoTracker_LSTCore_src_alpaka_Hit_h
#define RecoTracker_LSTCore_src_alpaka_Hit_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/alpakastdAlgorithm.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/HitsDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhiChange(TAcc const& acc, float x1, float y1, float x2, float y2) {
    return cms::alpakatools::deltaPhi(acc, x1, y1, x2 - x1, y2 - y1);
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
                                  uint16_t Endcap,          // Integer corresponding to endcap in module subdets
                                  uint16_t TwoS,            // Integer corresponding to TwoS in moduleType
                                  unsigned int nModules,    // Number of modules
                                  unsigned int nEndCapMap,  // Number of elements in endcap map
                                  EndcapGeometryDevConst endcapGeometry,
                                  ModulesConst modules,
                                  Hits hits,
                                  HitsRanges hitsRanges) const  // Total number of hits in event
    {
      auto geoMapDetId = endcapGeometry.geoMapDetId();  // DetId's from endcap map
      auto geoMapPhi = endcapGeometry.geoMapPhi();      // Phi values from endcap map
      unsigned int nHits = hits.metadata().size();
      for (unsigned int ihit : cms::alpakatools::uniform_elements(acc, nHits)) {
        float ihit_x = hits.xs()[ihit];
        float ihit_y = hits.ys()[ihit];
        float ihit_z = hits.zs()[ihit];
        int iDetId = hits.detid()[ihit];

        hits.rts()[ihit] = alpaka::math::sqrt(acc, ihit_x * ihit_x + ihit_y * ihit_y);
        hits.phis()[ihit] = cms::alpakatools::phi(acc, ihit_x, ihit_y);
        hits.etas()[ihit] =
            ((ihit_z > 0) - (ihit_z < 0)) *
            alpaka::math::acosh(
                acc, alpaka::math::sqrt(acc, ihit_x * ihit_x + ihit_y * ihit_y + ihit_z * ihit_z) / hits.rts()[ihit]);
        auto found_pointer = alpaka_std::lower_bound(modules.mapdetId(), modules.mapdetId() + nModules, iDetId);
        ALPAKA_ASSERT_ACC(found_pointer != modules.mapdetId() + nModules);
        int found_index = std::distance(modules.mapdetId(), found_pointer);
        uint16_t lastModuleIndex = modules.mapIdx()[found_index];

        hits.moduleIndices()[ihit] = lastModuleIndex;

        if (modules.subdets()[lastModuleIndex] == Endcap && modules.moduleType()[lastModuleIndex] == TwoS) {
          found_pointer = alpaka_std::lower_bound(geoMapDetId, geoMapDetId + nEndCapMap, iDetId);
          ALPAKA_ASSERT_ACC(found_pointer != geoMapDetId + nEndCapMap);
          found_index = std::distance(geoMapDetId, found_pointer);
          float phi = geoMapPhi[found_index];
          float cos_phi = alpaka::math::cos(acc, phi);
          hits.highEdgeXs()[ihit] = ihit_x + 2.5f * cos_phi;
          hits.lowEdgeXs()[ihit] = ihit_x - 2.5f * cos_phi;
          float sin_phi = alpaka::math::sin(acc, phi);
          hits.highEdgeYs()[ihit] = ihit_y + 2.5f * sin_phi;
          hits.lowEdgeYs()[ihit] = ihit_y - 2.5f * sin_phi;
        }
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
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
