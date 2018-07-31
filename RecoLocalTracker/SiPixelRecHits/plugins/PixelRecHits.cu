// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// thrust heders
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

// CMSSW headers
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"
#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace pixelgpudetails {
  PixelRecHitGPUKernel::PixelRecHitGPUKernel(cuda::stream_t<>& cudaStream) {

    cudaCheck(cudaMalloc((void**) & gpu_.bs_d,3*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.hitsModuleStart_d,(gpuClustering::MaxNumModules+1)*sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**) & gpu_.hitsLayerStart_d,(11)*sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**) & gpu_.charge_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.detInd_d,(gpuClustering::MaxNumModules*256)*sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**) & gpu_.xg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.yg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.zg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.rg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.xl_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.yl_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.xerr_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.yerr_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.iphi_d,(gpuClustering::MaxNumModules*256)*sizeof(int16_t)));
    cudaCheck(cudaMalloc((void**) & gpu_.sortIndex_d,(gpuClustering::MaxNumModules*256)*sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**) & gpu_.mr_d,(gpuClustering::MaxNumModules*256)*sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**) & gpu_.mc_d,(gpuClustering::MaxNumModules*256)*sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**) & gpu_.hist_d, 10*sizeof(HitsOnGPU::Hist)));
    cudaCheck(cudaMalloc((void**) & gpu_d, sizeof(HitsOnGPU)));
    gpu_.me_d = gpu_d;
    cudaCheck(cudaMemcpyAsync(gpu_d, &gpu_, sizeof(HitsOnGPU), cudaMemcpyDefault,cudaStream.id()));
  }

  PixelRecHitGPUKernel::~PixelRecHitGPUKernel() {
    cudaCheck(cudaFree(gpu_.hitsModuleStart_d));
    cudaCheck(cudaFree(gpu_.charge_d));
    cudaCheck(cudaFree(gpu_.detInd_d));
    cudaCheck(cudaFree(gpu_.xg_d));
    cudaCheck(cudaFree(gpu_.yg_d));
    cudaCheck(cudaFree(gpu_.zg_d));
    cudaCheck(cudaFree(gpu_.rg_d));
    cudaCheck(cudaFree(gpu_.xl_d));
    cudaCheck(cudaFree(gpu_.yl_d));
    cudaCheck(cudaFree(gpu_.xerr_d));
    cudaCheck(cudaFree(gpu_.yerr_d));
    cudaCheck(cudaFree(gpu_.iphi_d));
    cudaCheck(cudaFree(gpu_.sortIndex_d));
    cudaCheck(cudaFree(gpu_.mr_d));
    cudaCheck(cudaFree(gpu_.mc_d));
    cudaCheck(cudaFree(gpu_.hist_d));
    cudaCheck(cudaFree(gpu_d));
  }

  void PixelRecHitGPUKernel::makeHitsAsync(const siPixelRawToClusterHeterogeneousProduct::GPUProduct& input,
                                           float const * bs,
                                           pixelCPEforGPU::ParamsOnGPU const * cpeParams,
                                           cuda::stream_t<>& stream) {

   cudaCheck(cudaMemcpyAsync(gpu_.bs_d, bs, 3*sizeof(float), cudaMemcpyDefault, stream.id()));

    // Set first the first element to 0
    cudaCheck(cudaMemsetAsync(gpu_.hitsModuleStart_d, 0, sizeof(uint32_t), stream.id()));
    // Then use inclusive_scan to get the partial sum to the rest
    thrust::inclusive_scan(thrust::cuda::par.on(stream.id()),
                           input.clusInModule_d,
                           input.clusInModule_d + gpuClustering::MaxNumModules,
                           &gpu_.hitsModuleStart_d[1]);

    int threadsPerBlock = 256;
    int blocks = input.nModules; // active modules (with digis)
    gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, stream.id()>>>(
      cpeParams,
      gpu_.bs_d,
      input.moduleInd_d,
      input.xx_d, input.yy_d, input.adc_d,
      input.moduleStart_d,
      input.clusInModule_d, input.moduleId_d,
      input.clus_d,
      input.nDigis,
      gpu_.hitsModuleStart_d,
      gpu_.charge_d,
      gpu_.detInd_d,
      gpu_.xg_d, gpu_.yg_d, gpu_.zg_d, gpu_.rg_d,
      gpu_.iphi_d,
      gpu_.xl_d, gpu_.yl_d,
      gpu_.xerr_d, gpu_.yerr_d,
      gpu_.mr_d, gpu_.mc_d
    );

    // needed only if hits on CPU are required...
    cudaCheck(cudaMemcpyAsync(hitsModuleStart_, gpu_.hitsModuleStart_d, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t), cudaMemcpyDefault, stream.id()));

    // to be moved to gpu?
    auto nhits = hitsModuleStart_[gpuClustering::MaxNumModules];
    for (int i=0;i<10;++i) hitsLayerStart_[i]=hitsModuleStart_[phase1PixelTopology::layerStart[i]];
    hitsLayerStart_[10]=nhits;

#ifdef GPU_DEBUG
    std::cout << "hit layerStart ";
    for (int i=0;i<10;++i) std::cout << phase1PixelTopology::layerName[i] << ':' << hitsLayerStart_[i] << ' ';
    std::cout << "end:" << hitsLayerStart_[10] << std::endl;
#endif

    cudaCheck(cudaMemcpyAsync(gpu_.hitsLayerStart_d, hitsLayerStart_, (11) * sizeof(uint32_t), cudaMemcpyDefault, stream.id()));

    // for timing test
    // radixSortMultiWrapper<int16_t><<<10, 256, 0, c.stream>>>(gpu_.iphi_d,gpu_.sortIndex_d,gpu_.hitsLayerStart_d);

    cudautils::fillManyFromVector(gpu_.hist_d,10,gpu_.iphi_d, gpu_.hitsLayerStart_d, nhits,256,stream.id());
  }

  HitsOnCPU PixelRecHitGPUKernel::getOutput(cuda::stream_t<>& stream) const {
    // needed only if hits on CPU are required...
    auto nhits = hitsModuleStart_[gpuClustering::MaxNumModules];

    HitsOnCPU hoc(nhits);
    hoc.gpu_d = gpu_d;
    memcpy(hoc.hitsModuleStart, hitsModuleStart_, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t));
    cudaCheck(cudaMemcpyAsync(hoc.charge.data(), gpu_.charge_d, nhits*sizeof(int32_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.xl.data(), gpu_.xl_d, nhits*sizeof(float), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.yl.data(), gpu_.yl_d, nhits*sizeof(float), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.xe.data(), gpu_.xerr_d, nhits*sizeof(float), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.ye.data(), gpu_.yerr_d, nhits*sizeof(float), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.mr.data(), gpu_.mr_d, nhits*sizeof(uint16_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.mc.data(), gpu_.mc_d, nhits*sizeof(uint16_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaStreamSynchronize(stream.id()));
    return hoc;
  }
}
