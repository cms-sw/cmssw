#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include "RecoLocalTracker/SiPixelClusterizer/plugins/siPixelRawToClusterHeterogeneousProduct.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusteringConstants.h"

#include <cuda/api_wrappers.h>

#include <cstdint>
#include <vector>

#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h" 


namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

namespace pixelgpudetails {
  using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;

  using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

  class PixelRecHitGPUKernel {
  public:
    PixelRecHitGPUKernel(cuda::stream_t<>& cudaStream);
    ~PixelRecHitGPUKernel();

    PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
    PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

    void makeHitsAsync(const siPixelRawToClusterHeterogeneousProduct::GPUProduct& input,
                       float const * bs,
                       pixelCPEforGPU::ParamsOnGPU const * cpeParams,
                       bool transferToCPU,
                       cuda::stream_t<>& stream);

    HitsOnCPU getOutput() const {
      return HitsOnCPU{
        h_hitsModuleStart_, h_detInd_, h_charge_,
        h_xl_, h_yl_, h_xe_, h_ye_, h_mr_, h_mc_,
        gpu_d, nhits_
      };
    }

  private:
    HitsOnGPU * gpu_d;  // copy of the structure on the gpu itself: this is the "Product" 
    HitsOnGPU gpu_;
    uint32_t nhits_ = 0;
    uint32_t *d_phase1TopologyLayerStart_ = nullptr;
    uint8_t *d_phase1TopologyLayer_ = nullptr;
    uint32_t *h_hitsModuleStart_ = nullptr;
    uint16_t *h_detInd_ = nullptr;
    int32_t *h_charge_ = nullptr;
    float *h_xl_ = nullptr;
    float *h_yl_ = nullptr;
    float *h_xe_ = nullptr;
    float *h_ye_ = nullptr;
    uint16_t *h_mr_ = nullptr;
    uint16_t *h_mc_ = nullptr;
    void *h_owner_32bit_ = nullptr;
    size_t h_owner_32bit_pitch_ = 0;
    void *h_owner_16bit_ = nullptr;
    size_t h_owner_16bit_pitch_ = 0;
#ifdef GPU_DEBUG
    uint32_t *h_hitsLayerStart_ = nullptr;
#endif
  };
}

#endif // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
