#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHitGPUKernelT_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHitGPUKernelT_h

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
//#define GPU_DEBUG 1
namespace pixelgpudetails {

  template <typename TrackerTraits>
  class PixelRecHitGPUKernelT {
  public:
    PixelRecHitGPUKernelT() = default;
    ~PixelRecHitGPUKernelT() = default;

    PixelRecHitGPUKernelT(const PixelRecHitGPUKernelT&) = delete;
    PixelRecHitGPUKernelT(PixelRecHitGPUKernelT&&) = delete;
    PixelRecHitGPUKernelT& operator=(const PixelRecHitGPUKernelT&) = delete;
    PixelRecHitGPUKernelT& operator=(PixelRecHitGPUKernelT&&) = delete;

    using ParamsOnGPU = pixelCPEforGPU::ParamsOnGPUT<TrackerTraits>;

    TrackingRecHit2DGPUT<TrackerTraits> makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                                      SiPixelClustersCUDA const& clusters_d,
                                                      BeamSpotCUDA const& bs_d,
                                                      ParamsOnGPU const* cpeParams,
                                                      cudaStream_t stream) const;
  };

}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHitGPUKernelT_h
