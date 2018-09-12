#ifndef RecoLocalTracker_SiPixelRecHits_plugins_siPixelRecHitsHeterogeneousProduct_h
#define RecoLocalTracker_SiPixelRecHits_plugins_siPixelRecHitsHeterogeneousProduct_h

#include <cstdint>
#include <vector>

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

namespace siPixelRecHitsHeterogeneousProduct {

  using CPUProduct = int; // dummy

  struct HitsOnGPU{
     pixelCPEforGPU::ParamsOnGPU const * cpeParams = nullptr;    // forwarded from setup, NOT owned
     float * bs_d;
     const uint32_t * hitsModuleStart_d; // forwarded from clusters
     uint32_t * hitsLayerStart_d;
     int32_t  * charge_d;
     uint16_t * detInd_d;
     float *xg_d, *yg_d, *zg_d, *rg_d;
     float *xl_d, *yl_d;
     float *xerr_d, *yerr_d;
     int16_t * iphi_d;
     uint16_t * sortIndex_d;
     uint16_t * mr_d;
     uint16_t * mc_d;

     using Hist = HistoContainer<int16_t,7,8>;
     Hist * hist_d;

     HitsOnGPU const * me_d = nullptr;

    // Owning pointers to the 32/16 bit arrays with size MAX_HITS
    void *owner_32bit_;
    size_t owner_32bit_pitch_;
    void *owner_16bit_;
    size_t owner_16bit_pitch_;
  };

  struct HitsOnCPU {
    uint32_t const * hitsModuleStart = nullptr;
    uint16_t const * detInd = nullptr;
    int32_t const * charge = nullptr;
    float const * xl = nullptr;
    float const * yl = nullptr;
    float const * xe = nullptr;
    float const * ye = nullptr;
    uint16_t const * mr = nullptr;
    uint16_t const * mc = nullptr;

    HitsOnGPU const * gpu_d = nullptr;
    uint32_t nHits;
  };

  using GPUProduct = HitsOnCPU;  // FIXME fill cpu vectors on demand

  using HeterogeneousPixelRecHit = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                            heterogeneous::GPUCudaProduct<GPUProduct> >;
}

#endif // RecoLocalTracker_SiPixelRecHits_plugins_siPixelRecHitsHeterogeneousProduct_h
