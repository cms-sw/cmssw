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


  __global__ void getHits(pixelCPEforGPU::ParamsOnGPU const * cpeParams,
                          uint16_t const * id,
			  uint16_t const * x,
			  uint16_t const * y,
			  uint16_t const * adc,
			  uint32_t const * digiModuleStart,
			  uint32_t * const clusInModule,
			  uint32_t * const moduleId,
			  int32_t * const  clus,
			  int numElements,
			  uint32_t const * hitsModuleStart,
                          int32_t * chargeh,
			  float * xh, float * yh, float * zh,
			  bool local // if true fill just x & y in local coord...
			  ){

    // as usual one block per module

    __shared__ ClusParams clusParams;


    auto first = digiModuleStart[1 + blockIdx.x];  
    
    auto me = id[first];
    assert (moduleId[blockIdx.x]==me);

    auto nclus = clusInModule[me];

#ifdef GPU_DEBUG
    if (me%100==1)
      if (threadIdx.x==0) printf("hitbuilder: %d clusters in module %d. will write at %d\n",nclus,me,hitsModuleStart[me]);
#endif

    assert(blockDim.x>=MaxClusInModule);
    assert(nclus<=MaxClusInModule);

    auto ic = threadIdx.x;
    
    if (ic<nclus) {
      clusParams.minRow[ic] = std::numeric_limits<uint32_t>::max();
      clusParams.maxRow[ic] = 0;
      clusParams.minCol[ic] = std::numeric_limits<uint32_t>::max();
      clusParams.maxCol[ic] = 0;
      
      clusParams.charge[ic] = 0;

      clusParams.Q_f_X[ic] = 0;
      clusParams.Q_l_X[ic] = 0;
      clusParams.Q_f_Y[ic] = 0;
      clusParams.Q_l_Y[ic] = 0;
    }

    
    first+=threadIdx.x;
    
    __syncthreads();


    // one thead per "digi"
    
    for (int i=first; i<numElements; i+=blockDim.x) {
      if (id[i]==InvId) continue;  // not valid
      if (id[i]!=me) break;  // end of module
      assert(clus[i]<nclus);
      atomicMin(&clusParams.minRow[clus[i]],x[i]);
      atomicMax(&clusParams.maxRow[clus[i]],x[i]);
      atomicMin(&clusParams.minCol[clus[i]],y[i]);
      atomicMax(&clusParams.maxCol[clus[i]],y[i]);
    }

    __syncthreads();

    for (int i=first; i<numElements; i+=blockDim.x) {
      if (id[i]==InvId) continue;  // not valid
      if (id[i]!=me) break;  // end of module
      atomicAdd(&clusParams.charge[clus[i]],adc[i]);
      if (clusParams.minRow[clus[i]]==x[i]) atomicAdd(&clusParams.Q_f_X[clus[i]],adc[i]); 
      if (clusParams.maxRow[clus[i]]==x[i]) atomicAdd(&clusParams.Q_l_X[clus[i]],adc[i]); 
      if (clusParams.minCol[clus[i]]==y[i]) atomicAdd(&clusParams.Q_f_Y[clus[i]],adc[i]);
      if (clusParams.maxCol[clus[i]]==y[i]) atomicAdd(&clusParams.Q_l_Y[clus[i]],adc[i]); 
    }

    __syncthreads();

    // next one cluster per thread...
    if (ic>=nclus) return;

    first = hitsModuleStart[me];
    auto h = first+ic;  // output index in global memory

    assert(h<2000*256);

    pixelCPEforGPU::position(cpeParams->commonParams(), cpeParams->detParams(me), clusParams,ic);

    chargeh[h] = clusParams.charge[ic];

    if (local) {   
     xh[h]= clusParams.xpos[ic];   
     yh[h]= clusParams.ypos[ic]; 
    } else {
      cpeParams->detParams(me).frame.toGlobal(clusParams.xpos[ic],clusParams.ypos[ic],
                                              xh[h],yh[h],zh[h]
                                             );
    }
    
  }

}

