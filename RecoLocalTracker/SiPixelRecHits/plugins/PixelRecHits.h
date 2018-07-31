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
                       cuda::stream_t<>& stream);

    std::unique_ptr<HitsOnCPU>&& getOutput(cuda::stream_t<>& stream);

  private:
    HitsOnGPU * gpu_d;  // copy of the structure on the gpu itself: this is the "Product" 
    HitsOnGPU gpu_;
    std::unique_ptr<HitsOnCPU> cpu_;
    uint32_t *d_phase1TopologyLayerStart_ = nullptr;
    uint32_t hitsModuleStart_[gpuClustering::MaxNumModules+1];
    uint32_t hitsLayerStart_[11];
  };
}

#endif // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
