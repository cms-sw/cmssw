#include "RecoLocalTracker/SiPixelRecHits/interface/gpuPixelRecHits.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"

#include "EventFilter/SiPixelRawToDigi/plugins/RawToDigiGPU.h" // for context....

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <numeric>
#include <algorithm>


void pixelRecHits_wrapper(
      context const & c,
      pixelCPEforGPU::ParamsOnGPU const * cpeParams,
      uint32_t ndigis
)
{

 
 uint32_t hitsModuleStart[gpuClustering::MaxNumModules+1];
 hitsModuleStart[0] =0;
 cudaMemcpyAsync(&hitsModuleStart[1], c.clusInModule_d, gpuClustering::MaxNumModules*sizeof(uint32_t), cudaMemcpyDeviceToHost, c.stream); 

 std::partial_sum(std::begin(hitsModuleStart),std::end(hitsModuleStart),std::begin(hitsModuleStart));

 auto nhits = hitsModuleStart[gpuClustering::MaxNumModules];
 std::cout << " total number of clusters " << nhits << std::endl;

 float xg[nhits],yg[nhits],zg[1];
  
 int threadsPerBlock = 256;
 int blocks = gpuClustering::MaxNumModules;
 gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, c.stream>>>(
               cpeParams,
               c.moduleInd_d,
               c.xx_d, c.yy_d, c.adc_d,
               c.moduleStart_d,
               c.clusInModule_d, c.moduleId_d,
               c.clus_d,
               ndigis,
               hitsModuleStart,
               xg,yg,zg,
               false
  );

}


// aaaaggggghhhhhhh
struct TheGlobalByVin {
  static
  context const * & theContext() {
    static context const * me;
    return me;
  }    
  static 
  uint32_t & ndigis() {
    static uint32_t me;
    return me;
  }
};
//

void pixelRecHitsGlobal(pixelCPEforGPU::ParamsOnGPU const * cpeParams) {
  pixelRecHits_wrapper(*TheGlobalByVin::theContext(), cpeParams, TheGlobalByVin::ndigis());
}
