#pragma once

#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <cassert>


namespace gpuPixelRecHits {

  // to be moved in common namespace...
  constexpr uint16_t InvId=9999; // must be > MaxNumModules

  
  constexpr uint32_t MaxClusInModule=  pixelCPEforGPU::MaxClusInModule;

  using ClusParams = pixelCPEforGPU::ClusParams;


  __global__ void getHits(uint16_t const * id,
			  uint16_t const * x,
			  uint16_t const * y,
			  uint16_t const * adc,
			  uint32_t const * digiModuleStart,
			  uint32_t * const clusInModule,
			  uint32_t * const moduleId,
			  int32_t * const  clus,
			  int numElements,
			  uint32_t const * hitsModuleStart,
			  float * xh, float * yh, float * zh,
			  bool local // if true fill just x & y in local coord...
			  ){

    // as usual one block per module

    __shared__ ClusParams clusParams;


    auto first = digiModuleStart[1 + blockIdx.x];  
    
    auto me = id[first];
    assert (moduleId[blockIdx.x]==me);

    auto nclus = clusInModule[me];

    assert(blockDim.x>=MaxClusInModule);
    assert(nclus<=MaxClusInModule);

    auto i = threadIdx.x;
    
    if (threadIdx.x<nclus) {
      clusParams.minRow[i] = std::numeric_limits<uint32_t>::max();
      clusParams.maxRow[i] = 0;
      clusParams.minCol[i] = std::numeric_limits<uint32_t>::max();
      clusParams.maxCol[i] = 0;
      
      clusParams.Q_f_X[i] = 0;
      clusParams.Q_l_X[i] = 0;
      clusParams.Q_f_Y[i] = 0;
      clusParams.Q_l_Y[i] = 0;
    }

    
    first+=threadIdx.x;
    
    __syncthreads();


    // one thead per "digi"
    
    for (int i=first; i<numElements; i+=blockDim.x) {
      if (id[i]==InvId) continue;  // not valid
      if (id[i]!=me) break;  // end of module
      atomicMin(&clusParams.minRow[clus[i]],x[i]);
      atomicMax(&clusParams.maxRow[clus[i]],x[i]);
      atomicMin(&clusParams.minCol[clus[i]],y[i]);
      atomicMax(&clusParams.maxCol[clus[i]],y[i]);
    }

    __syncthreads();

    for (int i=first; i<numElements; i+=blockDim.x) {
      if (id[i]==InvId) continue;  // not valid
      if (id[i]!=me) break;  // end of module
      if (clusParams.minRow[clus[i]]==x[i]) atomicAdd(&clusParams.Q_f_X[clus[i]],adc[i]); 
      if (clusParams.maxRow[clus[i]]==x[i]) atomicAdd(&clusParams.Q_l_X[clus[i]],adc[i]); 
      if (clusParams.minCol[clus[i]]==y[i]) atomicAdd(&clusParams.Q_f_Y[clus[i]],adc[i]);
      if (clusParams.maxCol[clus[i]]==y[i]) atomicAdd(&clusParams.Q_l_Y[clus[i]],adc[i]); 
    }

    __syncthreads();

    // next one cluster per thread...
    if (threadIdx.x>=nclus) return;

    first = hitsModuleStart[me];
    auto h = first+i;  // output index in global memory

   
    xh[h]=0; // fake;   
     
    
  }

}

