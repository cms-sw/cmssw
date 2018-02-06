#pragma once

#include <cstdint>
#include <cstdio>

namespace gpuClustering {

  constexpr uint32_t MaxNumModules = 2000;

  constexpr uint32_t MaxNumPixels = 256*2000;  // this does not mean maxPixelPerModule==256!
  
  constexpr uint16_t InvId=9999; // must be > MaxNumModules
  
  __global__ void countModules(uint16_t const * id,
			       uint32_t * moduleStart,
			       int32_t * clus,
			       int numElements){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numElements) return;
    clus[i]=i;
    if (InvId==id[i]) return;
    auto j=i-1;
    while(j>=0 && id[j]==InvId) --j;
    if(j<0 || id[j]!=id[i]) {
      // boundary...
      auto loc = atomicInc(moduleStart,MaxNumModules);
      moduleStart[loc+1]=i;
    }
  }
  
  
  __global__ void findClus(uint16_t const * id,
			   uint16_t const * x,
			   uint16_t const * y,
			   uint8_t const * adc,
			   uint32_t const * moduleStart,
			 uint32_t * clusInModule, uint32_t * moduleId,
			   int32_t * clus, uint32_t * debug, 
			   int numElements){
    
    __shared__ bool go;
    __shared__ int nclus;
    
    
    auto first = moduleStart[1 + blockIdx.x];  
    
    auto me = id[first];
    
    first+=threadIdx.x;
    
    go=true;
    nclus=0;
    __syncthreads();
    
    
    while (go) {
      __syncthreads();
      go=false;
      __syncthreads();

      for (int i=first; i<numElements; i+=blockDim.x) {
	if (id[i]==InvId) continue;  // not valid
	if (id[i]!=me) break;  // end of module
	++debug[i];
	for (int j=i+1; j<numElements; ++j) {
	if (id[j]==InvId) continue;  // not valid
	if (id[j]!=me) break;  // end of module

	if (std::abs(int(x[j])-int(x[i]))>1) continue;
	if (std::abs(int(y[j])-int(y[i]))>1) continue;
	auto old = atomicMin(&clus[j],clus[i]);
	if (old!=clus[i]) go=true;
	atomicMin(&clus[i],old);
	}
      }
      
    __syncthreads();
    }
    
    /*
    // fast count (nice but not much useful)
    auto laneId = threadIdx.x & 0x1f;
    
    for (int i=first; i<numElements; i+=blockDim.x) {
    if (id[i]==InvId) continue;  // not valid
    if (id[i]!=me) break;  // end of module
    auto value = clus[i]^i;
    auto mask = __ballot_sync(0xffffffff,value==0);
    if (laneId==0) atomicAdd(&nclus,__popc(mask));
    }
    
    __syncthreads();
    if (me<30)
    if (laneId==0) printf("%d clusters in module %d\n",nclus,me);
    */

    nclus=0;
    __syncthreads();
    for (int i=first; i<numElements; i+=blockDim.x) {
      if (id[i]==InvId) continue;  // not valid
      if (id[i]!=me) break;  // end of module
      if (clus[i]==i) {
	auto old = atomicAdd(&nclus,1);
	clus[i]=-(old+1);
      }
    }
    
    __syncthreads();
    
    for (int i=first; i<numElements; i+=blockDim.x) {
      if (id[i]==InvId) continue;  // not valid
      if (id[i]!=me) break;  // end of module
      if (clus[i]>=0) clus[i]=clus[clus[i]];
    }
    
    __syncthreads();
    for (int i=first; i<numElements; i+=blockDim.x) {
      if (id[i]==InvId) {clus[i]=-9999; continue; } // not valid
      if (id[i]!=me) break;  // end of module
      clus[i] = -clus[i] -1;
    }
    
  
    __syncthreads();
    if (threadIdx.x==0) {
      clusInModule[blockIdx.x]=nclus;
      moduleId[blockIdx.x]=me;
    }
    
    
    if (me<30)
      if (threadIdx.x==0) printf("%d clusters in module %d\n",nclus,me);
    
  }
  
} //namespace gpuClustering

