#ifndef Hit_cuh
#define Hit_cuh

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#else
#include "Constants.h"
#include "Module.h"
#endif

namespace SDL {
  struct hits {
    unsigned int* nHits;
    float* xs;
    float* ys;
    float* zs;
    uint16_t* moduleIndices;
    unsigned int* idxs;
    unsigned int* detid;
    float* rts;
    float* phis;
    float* etas;
    float* highEdgeXs;
    float* highEdgeYs;
    float* lowEdgeXs;
    float* lowEdgeYs;
    int* hitRanges;
    int* hitRangesLower;
    int* hitRangesUpper;
    int8_t* hitRangesnLower;
    int8_t* hitRangesnUpper;

    template <typename TBuff>
    void setData(TBuff& hitsbuf) {
      nHits = alpaka::getPtrNative(hitsbuf.nHits_buf);
      xs = alpaka::getPtrNative(hitsbuf.xs_buf);
      ys = alpaka::getPtrNative(hitsbuf.ys_buf);
      zs = alpaka::getPtrNative(hitsbuf.zs_buf);
      moduleIndices = alpaka::getPtrNative(hitsbuf.moduleIndices_buf);
      idxs = alpaka::getPtrNative(hitsbuf.idxs_buf);
      detid = alpaka::getPtrNative(hitsbuf.detid_buf);
      rts = alpaka::getPtrNative(hitsbuf.rts_buf);
      phis = alpaka::getPtrNative(hitsbuf.phis_buf);
      etas = alpaka::getPtrNative(hitsbuf.etas_buf);
      highEdgeXs = alpaka::getPtrNative(hitsbuf.highEdgeXs_buf);
      highEdgeYs = alpaka::getPtrNative(hitsbuf.highEdgeYs_buf);
      lowEdgeXs = alpaka::getPtrNative(hitsbuf.lowEdgeXs_buf);
      lowEdgeYs = alpaka::getPtrNative(hitsbuf.lowEdgeYs_buf);
      hitRanges = alpaka::getPtrNative(hitsbuf.hitRanges_buf);
      hitRangesLower = alpaka::getPtrNative(hitsbuf.hitRangesLower_buf);
      hitRangesUpper = alpaka::getPtrNative(hitsbuf.hitRangesUpper_buf);
      hitRangesnLower = alpaka::getPtrNative(hitsbuf.hitRangesnLower_buf);
      hitRangesnUpper = alpaka::getPtrNative(hitsbuf.hitRangesnUpper_buf);
    }
  };

  template <typename TDev>
  struct hitsBuffer : hits {
    Buf<TDev, unsigned int> nHits_buf;
    Buf<TDev, float> xs_buf;
    Buf<TDev, float> ys_buf;
    Buf<TDev, float> zs_buf;
    Buf<TDev, uint16_t> moduleIndices_buf;
    Buf<TDev, unsigned int> idxs_buf;
    Buf<TDev, unsigned int> detid_buf;
    Buf<TDev, float> rts_buf;
    Buf<TDev, float> phis_buf;
    Buf<TDev, float> etas_buf;
    Buf<TDev, float> highEdgeXs_buf;
    Buf<TDev, float> highEdgeYs_buf;
    Buf<TDev, float> lowEdgeXs_buf;
    Buf<TDev, float> lowEdgeYs_buf;
    Buf<TDev, int> hitRanges_buf;
    Buf<TDev, int> hitRangesLower_buf;
    Buf<TDev, int> hitRangesUpper_buf;
    Buf<TDev, int8_t> hitRangesnLower_buf;
    Buf<TDev, int8_t> hitRangesnUpper_buf;

    template <typename TQueue, typename TDevAcc>
    hitsBuffer(unsigned int nModules, unsigned int nMaxHits, TDevAcc const& devAccIn, TQueue& queue)
        : nHits_buf(allocBufWrapper<unsigned int>(devAccIn, 1u, queue)),
          xs_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          ys_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          zs_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          moduleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, nMaxHits, queue)),
          idxs_buf(allocBufWrapper<unsigned int>(devAccIn, nMaxHits, queue)),
          detid_buf(allocBufWrapper<unsigned int>(devAccIn, nMaxHits, queue)),
          rts_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          phis_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          etas_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          highEdgeXs_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          highEdgeYs_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          lowEdgeXs_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          lowEdgeYs_buf(allocBufWrapper<float>(devAccIn, nMaxHits, queue)),
          hitRanges_buf(allocBufWrapper<int>(devAccIn, nModules * 2, queue)),
          hitRangesLower_buf(allocBufWrapper<int>(devAccIn, nModules, queue)),
          hitRangesUpper_buf(allocBufWrapper<int>(devAccIn, nModules, queue)),
          hitRangesnLower_buf(allocBufWrapper<int8_t>(devAccIn, nModules, queue)),
          hitRangesnUpper_buf(allocBufWrapper<int8_t>(devAccIn, nModules, queue)) {
      alpaka::memset(queue, hitRanges_buf, 0xff);
      alpaka::memset(queue, hitRangesLower_buf, 0xff);
      alpaka::memset(queue, hitRangesUpper_buf, 0xff);
      alpaka::memset(queue, hitRangesnLower_buf, 0xff);
      alpaka::memset(queue, hitRangesnUpper_buf, 0xff);
      alpaka::wait(queue);
    }
  };

  // Alpaka does not support log10 natively right now.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float temp_log10(TAcc const& acc, float val) {
    constexpr float ln10 = 2.302585093f;  // precomputed ln(10)
    return alpaka::math::log(acc, val) / ln10;
  };

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float eta(TAcc const& acc, float x, float y, float z) {
    float r3 = alpaka::math::sqrt(acc, x * x + y * y + z * z);
    float rt = alpaka::math::sqrt(acc, x * x + y * y);
    float eta = ((z > 0) - (z < 0)) * alpaka::math::acosh(acc, r3 / rt);
    return eta;
  };

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi_mpi_pi(TAcc const& acc, float x) {
    if (alpaka::math::abs(acc, x) <= float(M_PI))
      return x;

    constexpr float o2pi = 1.f / (2.f * float(M_PI));
    float n = alpaka::math::round(acc, x * o2pi);
    return x - n * float(2.f * float(M_PI));
  };

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi(TAcc const& acc, float x, float y) {
    return phi_mpi_pi(acc, float(M_PI) + alpaka::math::atan2(acc, -y, -x));
  };

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhi(TAcc const& acc, float x1, float y1, float x2, float y2) {
    float phi1 = phi(acc, x1, y1);
    float phi2 = phi(acc, x2, y2);
    return phi_mpi_pi(acc, (phi2 - phi1));
  };

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhiChange(TAcc const& acc, float x1, float y1, float x2, float y2) {
    return deltaPhi(acc, x1, y1, x2 - x1, y2 - y1);
  };

  // Alpaka: This function is not yet implemented directly in Alpaka.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float copysignf(float a, float b) {
    int sign_a = (a < 0) ? -1 : 1;
    int sign_b = (b < 0) ? -1 : 1;
    return sign_a * sign_b * a;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float calculate_dPhi(float phi1, float phi2) {
    // Calculate dPhi
    float dPhi = phi1 - phi2;

    // Normalize dPhi to be between -pi and pi
    if (dPhi > float(M_PI)) {
      dPhi -= 2 * float(M_PI);
    } else if (dPhi < -float(M_PI)) {
      dPhi += 2 * float(M_PI);
    }

    return dPhi;
  };

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int binary_search(const unsigned int* data,  // Array that we are searching over
                                                        unsigned int search_val,  // Value we want to find in data array
                                                        unsigned int ndata)       // Number of elements in data array
  {
    unsigned int low = 0;
    unsigned int high = ndata - 1;

    while (low <= high) {
      unsigned int mid = (low + high) / 2;
      unsigned int test_val = data[mid];
      if (test_val == search_val)
        return mid;
      else if (test_val > search_val)
        high = mid - 1;
      else
        low = mid + 1;
    }
    // Couldn't find search value in array.
    return -1;
  };

  struct moduleRangesKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::hits hitsInGPU,
                                  int const& nLowerModules) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int lowerIndex = globalThreadIdx[2]; lowerIndex < nLowerModules; lowerIndex += gridThreadExtent[2]) {
        uint16_t upperIndex = modulesInGPU.partnerModuleIndices[lowerIndex];
        if (hitsInGPU.hitRanges[lowerIndex * 2] != -1 && hitsInGPU.hitRanges[upperIndex * 2] != -1) {
          hitsInGPU.hitRangesLower[lowerIndex] = hitsInGPU.hitRanges[lowerIndex * 2];
          hitsInGPU.hitRangesUpper[lowerIndex] = hitsInGPU.hitRanges[upperIndex * 2];
          hitsInGPU.hitRangesnLower[lowerIndex] =
              hitsInGPU.hitRanges[lowerIndex * 2 + 1] - hitsInGPU.hitRanges[lowerIndex * 2] + 1;
          hitsInGPU.hitRangesnUpper[lowerIndex] =
              hitsInGPU.hitRanges[upperIndex * 2 + 1] - hitsInGPU.hitRanges[upperIndex * 2] + 1;
        }
      }
    }
  };

  struct hitLoopKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t Endcap,                  // Integer corresponding to endcap in module subdets
                                  uint16_t TwoS,                    // Integer corresponding to TwoS in moduleType
                                  unsigned int nModules,            // Number of modules
                                  unsigned int nEndCapMap,          // Number of elements in endcap map
                                  const unsigned int* geoMapDetId,  // DetId's from endcap map
                                  const float* geoMapPhi,           // Phi values from endcap map
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::hits hitsInGPU,
                                  unsigned int const& nHits) const  // Total number of hits in event
    {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
      for (unsigned int ihit = globalThreadIdx[2]; ihit < nHits; ihit += gridThreadExtent[2]) {
        float ihit_x = hitsInGPU.xs[ihit];
        float ihit_y = hitsInGPU.ys[ihit];
        float ihit_z = hitsInGPU.zs[ihit];
        int iDetId = hitsInGPU.detid[ihit];

        hitsInGPU.rts[ihit] = alpaka::math::sqrt(acc, ihit_x * ihit_x + ihit_y * ihit_y);
        hitsInGPU.phis[ihit] = SDL::phi(acc, ihit_x, ihit_y);
        hitsInGPU.etas[ihit] =
            ((ihit_z > 0) - (ihit_z < 0)) *
            alpaka::math::acosh(
                acc,
                alpaka::math::sqrt(acc, ihit_x * ihit_x + ihit_y * ihit_y + ihit_z * ihit_z) / hitsInGPU.rts[ihit]);
        int found_index = binary_search(modulesInGPU.mapdetId, iDetId, nModules);
        uint16_t lastModuleIndex = modulesInGPU.mapIdx[found_index];

        hitsInGPU.moduleIndices[ihit] = lastModuleIndex;

        if (modulesInGPU.subdets[lastModuleIndex] == Endcap && modulesInGPU.moduleType[lastModuleIndex] == TwoS) {
          found_index = binary_search(geoMapDetId, iDetId, nEndCapMap);
          float phi = geoMapPhi[found_index];
          float cos_phi = alpaka::math::cos(acc, phi);
          hitsInGPU.highEdgeXs[ihit] = ihit_x + 2.5f * cos_phi;
          hitsInGPU.lowEdgeXs[ihit] = ihit_x - 2.5f * cos_phi;
          float sin_phi = alpaka::math::sin(acc, phi);
          hitsInGPU.highEdgeYs[ihit] = ihit_y + 2.5f * sin_phi;
          hitsInGPU.lowEdgeYs[ihit] = ihit_y - 2.5f * sin_phi;
        }
        // Need to set initial value if index hasn't been seen before.
        int old = alpaka::atomicOp<alpaka::AtomicCas>(
            acc, &(hitsInGPU.hitRanges[lastModuleIndex * 2]), -1, static_cast<int>(ihit));
        // For subsequent visits, stores the min value.
        if (old != -1)
          alpaka::atomicOp<alpaka::AtomicMin>(acc, &hitsInGPU.hitRanges[lastModuleIndex * 2], static_cast<int>(ihit));

        alpaka::atomicOp<alpaka::AtomicMax>(acc, &hitsInGPU.hitRanges[lastModuleIndex * 2 + 1], static_cast<int>(ihit));
      }
    }
  };
}  // namespace SDL
#endif
