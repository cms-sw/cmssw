// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// thrust heders
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

// CMSSW headers
#include "EventFilter/SiPixelRawToDigi/plugins/RawToDigiGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"
#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"


HitsOnGPU allocHitsOnGPU() {
   HitsOnGPU hh;
   cudaCheck(cudaMalloc((void**) & hh.hitsModuleStart_d,(gpuClustering::MaxNumModules+1)*sizeof(uint32_t)));
   cudaCheck(cudaMalloc((void**) & hh.charge_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
   cudaCheck(cudaMalloc((void**) & hh.xg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
   cudaCheck(cudaMalloc((void**) & hh.yg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
   cudaCheck(cudaMalloc((void**) & hh.zg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
   cudaCheck(cudaMalloc((void**) & hh.xerr_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
   cudaCheck(cudaMalloc((void**) & hh.yerr_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
   cudaCheck(cudaMalloc((void**) & hh.mr_d,(gpuClustering::MaxNumModules*256)*sizeof(uint16_t)));
   cudaCheck(cudaDeviceSynchronize());

   return hh;
}


HitsOnCPU
pixelRecHits_wrapper(
      context const & c,
      pixelCPEforGPU::ParamsOnGPU const * cpeParams,
      uint32_t ndigis,
      uint32_t nModules, // active modules (with digis)
      HitsOnGPU & hh
)
{
  thrust::exclusive_scan(thrust::cuda::par,
      c.clusInModule_d,
      c.clusInModule_d + gpuClustering::MaxNumModules + 1,
      hh.hitsModuleStart_d);
  
  int threadsPerBlock = 256;
  int blocks = nModules;
  gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, c.stream>>>(
      cpeParams,
      c.moduleInd_d,
      c.xx_d, c.yy_d, c.adc_d,
      c.moduleStart_d,
      c.clusInModule_d, c.moduleId_d,
      c.clus_d,
      ndigis,
      hh.hitsModuleStart_d,
      hh.charge_d,
      hh.xg_d, hh.yg_d, hh.zg_d,
      hh.xerr_d, hh.yerr_d, hh.mr_d,
      true // for the time being stay local...
      );


  // all this needed only if hits on CPU are required...
  uint32_t hitsModuleStart[gpuClustering::MaxNumModules+1];
  cudaCheck(cudaMemcpyAsync(hitsModuleStart, hh.hitsModuleStart_d, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t), cudaMemcpyDefault, c.stream));
  cudaCheck(cudaDeviceSynchronize());
  auto nhits = hitsModuleStart[gpuClustering::MaxNumModules];

  HitsOnCPU hoc(nhits);
  memcpy(hoc.hitsModuleStart, hitsModuleStart, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t));
  cudaCheck(cudaMemcpyAsync(hoc.charge.data(), hh.charge_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, c.stream));
  cudaCheck(cudaMemcpyAsync(hoc.xl.data(), hh.xg_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, c.stream));
  cudaCheck(cudaMemcpyAsync(hoc.yl.data(), hh.yg_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, c.stream));
  cudaCheck(cudaMemcpyAsync(hoc.xe.data(), hh.xerr_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, c.stream));
  cudaCheck(cudaMemcpyAsync(hoc.ye.data(), hh.yerr_d, nhits*sizeof(uint32_t), cudaMemcpyDefault, c.stream));
  cudaCheck(cudaMemcpyAsync(hoc.mr.data(), hh.mr_d, nhits*sizeof(uint16_t), cudaMemcpyDefault, c.stream));
  cudaCheck(cudaDeviceSynchronize());

  return hoc;
}
