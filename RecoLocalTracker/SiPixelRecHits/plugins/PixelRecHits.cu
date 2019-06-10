// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// CMSSW headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusteringConstants.h"

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace {
  __global__ void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                    uint32_t* hitsLayerStart) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    assert(0 == hitsModuleStart[0]);

    if (i < 11) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d %d: %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

  TrackingRecHit2DCUDA PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                                           SiPixelClustersCUDA const& clusters_d,
                                                           BeamSpotCUDA const& bs_d,
                                                           pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                           cuda::stream_t<>& stream) const {
    auto nHits = clusters_d.nClusters();
    TrackingRecHit2DCUDA hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), stream);

    int threadsPerBlock = 256;
    int blocks = digis_d.nModules();  // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
    if (blocks)  // protect from empty events
      gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, stream.id()>>>(cpeParams,
                                                                            bs_d.data(),
                                                                            digis_d.moduleInd(),
                                                                            digis_d.xx(),
                                                                            digis_d.yy(),
                                                                            digis_d.adc(),
                                                                            clusters_d.moduleStart(),
                                                                            clusters_d.clusInModule(),
                                                                            clusters_d.moduleId(),
                                                                            digis_d.clus(),
                                                                            digis_d.nDigis(),
                                                                            clusters_d.clusModuleStart(),
                                                                            hits_d.view());
    cudaCheck(cudaGetLastError());

    // assuming full warp of threads is better than a smaller number...
    setHitsLayerStart<<<1, 32, 0, stream.id()>>>(clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart());
    cudaCheck(cudaGetLastError());

    if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
      edm::LogWarning("PixelRecHitGPUKernel")
          << "Hits Overflow " << nHits << " > " << TrackingRecHit2DSOAView::maxHits();
    }

    if (nHits) {
      edm::Service<CUDAService> cs;
      auto hws = cs->make_device_unique<uint8_t[]>(TrackingRecHit2DSOAView::Hist::wsSize(), stream);
      cudautils::fillManyFromVector(
          hits_d.phiBinner(), hws.get(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256, stream.id());
      cudaCheck(cudaGetLastError());
    }
    return hits_d;
  }

}  // namespace pixelgpudetails
