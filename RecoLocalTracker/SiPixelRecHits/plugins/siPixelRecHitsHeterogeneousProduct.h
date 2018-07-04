#ifndef RecoLocalTracker_SiPixelRecHits_plugins_siPixelRecHitsHeterogeneousProduct_h
#define RecoLocalTracker_SiPixelRecHits_plugins_siPixelRecHitsHeterogeneousProduct_h

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"


#include <cstdint>
#include <vector>

// #include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"


namespace siPixelRecHitsHeterogeneousProduct {

  struct CPUProduct {
    SiPixelRecHitCollectionNew collection;
  };

  struct HitsOnGPU{
     float * bs_d;
     uint32_t * hitsModuleStart_d;
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

     // using Hist = HistoContainer<int16_t,7,8>;
     // Hist * hist_d;
  };


  struct HitsOnCPU {
    HitsOnCPU() = default;
    explicit HitsOnCPU(uint32_t nhits) :
      charge(nhits),xl(nhits),yl(nhits),xe(nhits),ye(nhits), mr(nhits), mc(nhits){}
    uint32_t hitsModuleStart[2001];
    std::vector<int32_t> charge;
    std::vector<float> xl, yl;
    std::vector<float> xe, ye;
    std::vector<uint16_t> mr;
    std::vector<uint16_t> mc;

    HitsOnGPU const * gpu_d=nullptr;
  };


  using GPUProduct = HitsOnCPU;  // FIXME fill cpu vectors on demand


  using HeterogeneousPixelRecHit = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                            heterogeneous::GPUCudaProduct<GPUProduct> >;


}



#endif
