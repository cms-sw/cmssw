#ifndef HeterogeneousCore_CUDAUtilities_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_prefixScan_h

#include <cstdint>

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

template<typename T>
__device__
void 
__forceinline__
warpPrefixScan(T * c, uint32_t i, uint32_t mask) {
   auto x = c[i];
   auto laneId = threadIdx.x & 0x1f;
   #pragma unroll
   for( int offset = 1 ; offset < 32 ; offset <<= 1 ) {
     auto y = __shfl_up_sync(mask,x, offset);
     if(laneId >= offset) x += y;
   }
   c[i] = x;
}

// limited to 32*32 elements....
template<typename T>
__device__
void
__forceinline__
blockPrefixScan(T * c, uint32_t size, T* ws) {
  assert(size<=1024);
  assert(0==blockDim.x%32);

  auto first = threadIdx.x;
  auto mask = __ballot_sync(0xffffffff,first<size);

  for (auto i=first; i<size; i+=blockDim.x) {
    warpPrefixScan(c,i,mask);
    auto laneId = threadIdx.x & 0x1f;
    auto warpId = i/32;
    assert(warpId<32);
    if (31==laneId) ws[warpId]=c[i];
    mask = __ballot_sync(mask,i+blockDim.x<size);
  }
  __syncthreads();
  if (size<=32) return;
  if (threadIdx.x<32) warpPrefixScan(ws,threadIdx.x,0xffffffff);
  __syncthreads();
  for (auto i=first+32; i<size; i+=blockDim.x) {
    auto warpId = i/32;
    c[i]+=ws[warpId-1];
  }
  __syncthreads();
}


#endif
