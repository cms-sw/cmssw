#ifndef RecoTracker_LSTCore_src_alpaka_Hit_h
#define RecoTracker_LSTCore_src_alpaka_Hit_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/HitsDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float eta(TAcc const& acc, float x, float y, float z) {
    float r3 = alpaka::math::sqrt(acc, x * x + y * y + z * z);
    float rt = alpaka::math::sqrt(acc, x * x + y * y);
    float eta = ((z > 0) - (z < 0)) * alpaka::math::acosh(acc, r3 / rt);
    return eta;
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi_mpi_pi(TAcc const& acc, float x) {
    if (alpaka::math::abs(acc, x) <= kPi)
      return x;

    constexpr float o2pi = 1.f / (2.f * kPi);
    float n = alpaka::math::round(acc, x * o2pi);
    return x - n * float(2.f * kPi);
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi(TAcc const& acc, float x, float y) {
    return phi_mpi_pi(acc, kPi + alpaka::math::atan2(acc, -y, -x));
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhi(TAcc const& acc, float x1, float y1, float x2, float y2) {
    float phi1 = phi(acc, x1, y1);
    float phi2 = phi(acc, x2, y2);
    return phi_mpi_pi(acc, (phi2 - phi1));
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhiChange(TAcc const& acc, float x1, float y1, float x2, float y2) {
    return deltaPhi(acc, x1, y1, x2 - x1, y2 - y1);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float calculate_dPhi(float phi1, float phi2) {
    // Calculate dPhi
    float dPhi = phi1 - phi2;

    // Normalize dPhi to be between -pi and pi
    if (dPhi > kPi) {
      dPhi -= 2 * kPi;
    } else if (dPhi < -kPi) {
      dPhi += 2 * kPi;
    }

    return dPhi;
  }

  struct ModuleRangesKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  HitsRanges hitsRanges,
                                  int nLowerModules) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int lowerIndex = globalThreadIdx[2]; lowerIndex < nLowerModules; lowerIndex += gridThreadExtent[2]) {
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
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t Endcap,          // Integer corresponding to endcap in module subdets
                                  uint16_t TwoS,            // Integer corresponding to TwoS in moduleType
                                  unsigned int nModules,    // Number of modules
                                  unsigned int nEndCapMap,  // Number of elements in endcap map
                                  EndcapGeometryDevConst endcapGeometry,
                                  ModulesConst modules,
                                  Hits hits,
                                  HitsRanges hitsRanges,
                                  unsigned int nHits) const  // Total number of hits in event
    {
      auto geoMapDetId = endcapGeometry.geoMapDetId();  // DetId's from endcap map
      auto geoMapPhi = endcapGeometry.geoMapPhi();      // Phi values from endcap map
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
      for (unsigned int ihit = globalThreadIdx[2]; ihit < nHits; ihit += gridThreadExtent[2]) {
        float ihit_x = hits.xs()[ihit];
        float ihit_y = hits.ys()[ihit];
        float ihit_z = hits.zs()[ihit];
        int iDetId = hits.detid()[ihit];

        hits.rts()[ihit] = alpaka::math::sqrt(acc, ihit_x * ihit_x + ihit_y * ihit_y);
        hits.phis()[ihit] = phi(acc, ihit_x, ihit_y);
        hits.etas()[ihit] =
            ((ihit_z > 0) - (ihit_z < 0)) *
            alpaka::math::acosh(
                acc, alpaka::math::sqrt(acc, ihit_x * ihit_x + ihit_y * ihit_y + ihit_z * ihit_z) / hits.rts()[ihit]);
        auto found_pointer = std::lower_bound(modules.mapdetId(), modules.mapdetId() + nModules, iDetId);
        int found_index = std::distance(modules.mapdetId(), found_pointer);
        if (found_pointer == modules.mapdetId() + nModules)
          found_index = -1;
        uint16_t lastModuleIndex = modules.mapIdx()[found_index];

        hits.moduleIndices()[ihit] = lastModuleIndex;

        if (modules.subdets()[lastModuleIndex] == Endcap && modules.moduleType()[lastModuleIndex] == TwoS) {
          found_pointer = std::lower_bound(geoMapDetId, geoMapDetId + nEndCapMap, iDetId);
          found_index = std::distance(geoMapDetId, found_pointer);
          if (found_pointer == geoMapDetId + nEndCapMap)
            found_index = -1;
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
