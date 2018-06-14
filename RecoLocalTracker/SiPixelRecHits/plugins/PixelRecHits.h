#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include "RecoLocalTracker/SiPixelClusterizer/plugins/siPixelRawToClusterHeterogeneousProduct.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusteringConstants.h"

#include <cuda/api_wrappers.h>

#include <cstdint>
#include <vector>

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

namespace pixelgpudetails {
  struct HitsOnGPU{
    uint32_t * hitsModuleStart_d;
    int32_t  * charge_d;
    float *xg_d, *yg_d, *zg_d;
    float *xerr_d, *yerr_d;
    uint16_t * mr_d;
  };

  struct HitsOnCPU {
    explicit HitsOnCPU(uint32_t nhits) :
      charge(nhits),xl(nhits),yl(nhits),xe(nhits),ye(nhits), mr(nhits){}
    uint32_t hitsModuleStart[2001];
    std::vector<int32_t> charge;
    std::vector<float> xl, yl;
    std::vector<float> xe, ye;
    std::vector<uint16_t> mr;
  };

  class PixelRecHitGPUKernel {
  public:
    PixelRecHitGPUKernel();
    ~PixelRecHitGPUKernel();

    PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
    PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

    void makeHitsAsync(const siPixelRawToClusterHeterogeneousProduct::GPUProduct& input,
                       pixelCPEforGPU::ParamsOnGPU const * cpeParams,
                       cuda::stream_t<>& stream);

    HitsOnCPU getOutput(cuda::stream_t<>& stream) const;

  private:
    HitsOnGPU gpu_;
    uint32_t hitsModuleStart_[gpuClustering::MaxNumModules+1];
  };
}

#endif // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
