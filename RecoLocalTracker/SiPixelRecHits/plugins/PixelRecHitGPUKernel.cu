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

// #define GPU_DEBUG

namespace {
  __global__ void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                    uint32_t* hitsLayerStart) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto m =
        cpeParams->commonParams().isPhase2 ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;

    assert(0 == hitsModuleStart[0]);

    if (i <= m) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d/%d at module %d: %d\n", i, m, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

  TrackingRecHit2DGPU PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                                          SiPixelClustersCUDA const& clusters_d,
                                                          BeamSpotCUDA const& bs_d,
                                                          pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                          bool isPhase2,
                                                          cudaStream_t stream) const {
    auto nHits = clusters_d.nClusters();

    TrackingRecHit2DGPU hits_d(
        nHits, isPhase2, clusters_d.offsetBPIX2(), cpeParams, clusters_d.clusModuleStart(), memoryPool::onDevice, stream);

    assert(hits_d.view());
    assert(hits_d.nMaxModules() == isPhase2 ? phase2PixelTopology::numberOfModules
                                            : phase1PixelTopology::numberOfModules);
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
      cudaCheck(cudaDeviceSynchronize());
#endif

    int activeModulesWithDigis = digis_d.nModules();
    // protect from empty events
    if (activeModulesWithDigis) {
      int threadsPerBlock = 128;
      int blocks = activeModulesWithDigis;

#ifdef GPU_DEBUG
      std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
      gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, stream>>>(
          cpeParams, bs_d.data(), digis_d.view(), digis_d.nDigis(), clusters_d.view(), hits_d.view());
      cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
      cudaCheck(cudaDeviceSynchronize());
#endif

      // assuming full warp of threads is better than a smaller number...
      if (nHits) {
        setHitsLayerStart<<<1, 32, 0, stream>>>(clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart());
        cudaCheck(cudaGetLastError());
        auto nLayers = isPhase2 ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;
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

}  // namespace pixelgpudetails
