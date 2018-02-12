#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"

#include "EventFilter/SiPixelRawToDigi/plugins/RawToDigiGPU.h" // for context....
#include "EventFilter/SiPixelRawToDigi/plugins/cudaCheck.h"

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <numeric>
#include <algorithm>

HitsOnGPU allocHitsOnGPU() {
   HitsOnGPU hh;
   cudaCheck(cudaMalloc((void**) & hh.hitsModuleStart_d,(gpuClustering::MaxNumModules+1)*sizeof(uint32_t)));
   cudaCheck(cudaMalloc((void**) & hh.xg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
   cudaCheck(cudaMalloc((void**) & hh.yg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
   cudaCheck(cudaMalloc((void**) & hh.zg_d,(gpuClustering::MaxNumModules*256)*sizeof(float)));
   cudaDeviceSynchronize();

   return hh;
}


void pixelRecHits_wrapper(
      context const & c,
      pixelCPEforGPU::ParamsOnGPU const * cpeParams,
      uint32_t ndigis,
      uint32_t nModules, // active modules (with digis)
      HitsOnGPU & hh
)
{

 
 uint32_t hitsModuleStart[gpuClustering::MaxNumModules+1];
 hitsModuleStart[0] =0;
 cudaCheck(cudaMemcpyAsync(&hitsModuleStart[1], c.clusInModule_d, gpuClustering::MaxNumModules*sizeof(uint32_t), cudaMemcpyDeviceToHost, c.stream)); 

 std::partial_sum(std::begin(hitsModuleStart),std::end(hitsModuleStart),std::begin(hitsModuleStart));

 auto nhits = hitsModuleStart[gpuClustering::MaxNumModules];
 std::cout << " total number of clusters " << nhits << std::endl;

 cudaCheck(cudaMemcpyAsync(hh.hitsModuleStart_d, &hitsModuleStart, (gpuClustering::MaxNumModules+1)*sizeof(uint32_t), cudaMemcpyHostToDevice, c.stream));

  
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
               hh.xg_d,hh.yg_d,hh.zg_d,
               false
  );

}
