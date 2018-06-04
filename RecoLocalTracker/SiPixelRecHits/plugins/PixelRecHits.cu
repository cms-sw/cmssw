// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// thrust heders
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

// CMSSW headers
#include "EventFilter/SiPixelRawToDigi/plugins/SiPixelRawToClusterGPUKernel.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"
#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace pixelgpudetails {
  PixelRecHitGPUKernel::PixelRecHitGPUKernel() {
    cudaCheck(cudaMalloc((void**) & gpu_.hitsModuleStart_d,(gpuClustering::MaxNumModules+1)*sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**) & gpu_.charge_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.xg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.yg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.zg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.xerr_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.yerr_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
    cudaCheck(cudaMalloc((void**) & gpu_.mr_d,(gpuClustering::MaxNumModules*256)*sizeof(uint16_t)));
  }

  PixelRecHitGPUKernel::~PixelRecHitGPUKernel() {
    cudaCheck(cudaFree(gpu_.hitsModuleStart_d));
    cudaCheck(cudaFree(gpu_.charge_d));
    cudaCheck(cudaFree(gpu_.xg_d));
    cudaCheck(cudaFree(gpu_.yg_d));
    cudaCheck(cudaFree(gpu_.zg_d));
    cudaCheck(cudaFree(gpu_.xerr_d));
    cudaCheck(cudaFree(gpu_.yerr_d));
    cudaCheck(cudaFree(gpu_.mr_d));
  }

  void PixelRecHitGPUKernel::makeHitsAsync(const siPixelRawToClusterHeterogeneousProduct::GPUProduct& input,
                                           pixelCPEforGPU::ParamsOnGPU const * cpeParams,
                                           cuda::stream_t<>& stream) {
    thrust::exclusive_scan(thrust::cuda::par,
                           input.clusInModule_d,
                           input.clusInModule_d + gpuClustering::MaxNumModules + 1,
                           gpu_.hitsModuleStart_d);
  
    int threadsPerBlock = 256;
    int blocks = input.nModules; // active modules (with digis)
    gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, stream.id()>>>(
      cpeParams,
      input.moduleInd_d,
      input.xx_d, input.yy_d, input.adc_d,
      input.moduleStart_d,
      input.clusInModule_d, input.moduleId_d,
      input.clus_d,
      input.nDigis,
      gpu_.hitsModuleStart_d,
      gpu_.charge_d,
      gpu_.xg_d, gpu_.yg_d, gpu_.zg_d,
      gpu_.xerr_d, gpu_.yerr_d, gpu_.mr_d,
      true // for the time being stay local...
    );

    // needed only if hits on CPU are required...
    cudaCheck(cudaMemcpyAsync(hitsModuleStart_, gpu_.hitsModuleStart_d, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
  }

  HitsOnCPU PixelRecHitGPUKernel::getOutput(cuda::stream_t<>& stream) const {
    // needed only if hits on CPU are required...
    auto nhits = hitsModuleStart_[gpuClustering::MaxNumModules];

    HitsOnCPU hoc(nhits);
    memcpy(hoc.hitsModuleStart, hitsModuleStart_, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t));
    cudaCheck(cudaMemcpyAsync(hoc.charge.data(), gpu_.charge_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.xl.data(), gpu_.xg_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.yl.data(), gpu_.yg_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.xe.data(), gpu_.xerr_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.ye.data(), gpu_.yerr_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(hoc.mr.data(), gpu_.mr_d, nhits*sizeof(uint16_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaStreamSynchronize(stream.id()));
    return hoc;
  }
}
