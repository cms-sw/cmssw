// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// CMSSW headers
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "PixelRecHitGPUKernel.h"
#include "gpuPixelRecHits.h"
// #define GPU_DEBUG 1

namespace {
  template <typename TrackerTraits>
  __global__ void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* cpeParams,
                                    uint32_t* hitsLayerStart) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr auto m = TrackerTraits::numberOfLayers;

    assert(0 == hitsModuleStart[0]);

    if (i <= m) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      int old = i == 0 ? 0 : hitsModuleStart[cpeParams->layerGeometry().layerStart[i - 1]];
      printf("LayerStart %d/%d at module %d: %d - %d\n",
             i,
             m,
             cpeParams->layerGeometry().layerStart[i],
             hitsLayerStart[i],
             hitsLayerStart[i] - old);
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

  template <typename TrackerTraits>
  TrackingRecHit2DGPUT<TrackerTraits> PixelRecHitGPUKernel<TrackerTraits>::makeHitsAsync(
      SiPixelDigisCUDA const& digis_d,
      SiPixelClustersCUDA const& clusters_d,
      BeamSpotCUDA const& bs_d,
      pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* cpeParams,
      cudaStream_t stream) const {
    using namespace gpuPixelRecHits;
    auto nHits = clusters_d.nClusters();

    TrackingRecHit2DGPUT<TrackerTraits> hits_d(
        nHits, clusters_d.offsetBPIX2(), cpeParams, clusters_d.clusModuleStart(), stream);

    int activeModulesWithDigis = digis_d.nModules();
    // protect from empty events
    if (activeModulesWithDigis) {
      int threadsPerBlock = 128;
      int blocks = activeModulesWithDigis;

#ifdef GPU_DEBUG

      std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
      getHits<TrackerTraits><<<blocks, threadsPerBlock, 0, stream>>>(
          cpeParams, bs_d.data(), digis_d.view(), digis_d.nDigis(), clusters_d.view(), hits_d.view());
      cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
      cudaCheck(cudaDeviceSynchronize());
#endif

      // assuming full warp of threads is better than a smaller number...
      if (nHits) {
        setHitsLayerStart<TrackerTraits>
            <<<1, 32, 0, stream>>>(clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart());
        cudaCheck(cudaGetLastError());
        constexpr auto nLayers = TrackerTraits::numberOfLayers;
        cms::cuda::fillManyFromVector(hits_d.phiBinner(),
                                      nLayers,
                                      hits_d.iphi(),
                                      hits_d.hitsLayerStart(),
                                      nHits,
                                      256,
                                      hits_d.phiBinnerStorage(),
                                      stream);
        cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
        cudaCheck(cudaDeviceSynchronize());
#endif
      }
    }

    return hits_d;
  }

  template class PixelRecHitGPUKernel<pixelTopology::Phase1>;
  template class PixelRecHitGPUKernel<pixelTopology::Phase2>;
}  // namespace pixelgpudetails
