#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHitGPUKernel_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHitGPUKernel_h

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
//#define GPU_DEBUG
namespace pixelgpudetails {

  template <typename TrackerTraits>
  class PixelRecHitGPUKernel {
  public:
    PixelRecHitGPUKernel() = default;
    ~PixelRecHitGPUKernel() = default;

    PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
    PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

    using ParamsOnGPU = pixelCPEforGPU::ParamsOnGPUT<TrackerTraits>;

    TrackingRecHitSoADevice<TrackerTraits> makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                                         SiPixelClustersCUDA const& clusters_d,
                                                         BeamSpotCUDA const& bs_d,
                                                         ParamsOnGPU const* cpeParams,
                                                         cudaStream_t stream) const;
  };

}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHitGPUKernel_h
