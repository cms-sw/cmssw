#ifndef HeterogeneousCoreCUDAUtilities_radixSort_H
#define HeterogeneousCoreCUDAUtilities_radixSort_H 

#include<cstdint>
#include<cassert>

template<typename T>
__device__  
void radixSort(T * a, uint16_t * ind, uint32_t size) {
    
  constexpr int d = 8, w = 8*sizeof(T);
  constexpr int sb = 1<<d;

  constexpr int MaxSize = 256*32;
  __shared__ uint16_t ind2[MaxSize];
  __shared__ int32_t c[sb], ct[sb], cu[sb];
  __shared__ uint32_t firstNeg;    

  __shared__ int ibs;
  __shared__ int p;

  assert(size>0);
  assert(size<=MaxSize); 
  assert(blockDim.x==sb);  

  // bool debug = false; // threadIdx.x==0 && blockIdx.x==5;

  firstNeg=0;

  p = 0;  

  auto j = ind;
  auto k = ind2;

  int32_t first = threadIdx.x;
  for (auto i=first; i<size; i+=blockDim.x)  j[i]=i;
  __syncthreads();


  while(p < w/d) {
    c[threadIdx.x]=0;
    __syncthreads();

    // fill bins
    for (auto i=first; i<size; i+=blockDim.x) {
      auto bin = (a[j[i]] >> d*p)&(sb-1);
      atomicAdd(&c[bin],1);
    }
    __syncthreads();

    // prefix scan "optimized"???...
    auto x = c[threadIdx.x];
    auto laneId = threadIdx.x & 0x1f;
    #pragma unroll
    for( int offset = 1 ; offset < 32 ; offset <<= 1 ) {
      auto y = __shfl_up_sync(0xffffffff,x, offset);
      if(laneId >= offset) x += y;
    }
    ct[threadIdx.x] = x;
    __syncthreads();
    auto ss = (threadIdx.x/32)*32 -1;
    c[threadIdx.x] = ct[threadIdx.x];
    for(int i=ss; i>0; i-=32) c[threadIdx.x] +=ct[i]; 
 
    /* 
    //prefix scan for the nulls  (for documentation)
    if (threadIdx.x==0)
      for (int i = 1; i < sb; ++i) c[i] += c[i-1];
    */

    
     // broadcast
     ibs =size-1;
     __syncthreads();
     while (ibs>0) {
       int i = ibs - threadIdx.x;
       cu[threadIdx.x]=-1;
       ct[threadIdx.x]=-1;
       __syncthreads();
       int32_t bin = -1;
       if (i>=0) { 
         bin = (a[j[i]] >> d*p)&(sb-1);
         ct[threadIdx.x]=bin;
         atomicMax(&cu[bin],int(i));
       }
       __syncthreads();
       if (i>=0 && i==cu[bin])  // ensure to keep them in order
         for (int ii=threadIdx.x; ii<blockDim.x; ++ii) if (ct[ii]==bin) {
           auto oi = ii-threadIdx.x; 
           // assert(i>=oi);if(i>=oi) 
           k[--c[bin]] = j[i-oi]; 
         }
       __syncthreads();
       if (bin>=0) assert(c[bin]>=0);
       if (threadIdx.x==0) ibs-=blockDim.x;
       __syncthreads();
     }    
      
    /*
    // broadcast for the nulls  (for documentation)
    if (threadIdx.x==0)
    for (int i=size-first-1; i>=0; i--) { // =blockDim.x) {
      auto bin = (a[j[i]] >> d*p)&(sb-1);
      auto ik = atomicSub(&c[bin],1);
      k[ik-1] = j[i];
    }
    */

    __syncthreads();
    assert(c[0]==0);


    // swap (local, ok)
    auto t=j;j=k;k=t;

    if (threadIdx.x==0) ++p;
    __syncthreads();
 
  }

  // w/d is even so ind is correct
  assert(j==ind);
  __syncthreads();

  

  // now move negative first...
  // find first negative  (for float ^ will not work...)
  for (auto i=first; i<size-1; i+=blockDim.x) {
    // if ( (int(a[ind[i]])*int(a[ind[i+1]])) <0 ) firstNeg=i+1;
   if ( (a[ind[i]]^a[ind[i+1]]) < 0 ) firstNeg=i+1; 
  }
  
  __syncthreads();
  // assert(firstNeg>0); not necessary true if all positive !

  auto ii=first;
  for (auto i=firstNeg+threadIdx.x; i<size; i+=blockDim.x)  { ind2[ii] = ind[i]; ii+=blockDim.x; }
  __syncthreads();
  ii= size-firstNeg +threadIdx.x;
  assert(ii>=0);
  for (auto i=first;i<firstNeg;i+=blockDim.x)  { ind2[ii] = ind[i]; ii+=blockDim.x; }
  __syncthreads();
  for (auto i=first; i<size; i+=blockDim.x) ind[i]=ind2[i];

  
}


template<typename T>
__device__
void radixSortMulti(T * v, uint16_t * index, uint32_t * offsets) {

  auto a = v+offsets[blockIdx.x];
  auto ind = index+offsets[blockIdx.x];;
  auto size = offsets[blockIdx.x+1]-offsets[blockIdx.x];
  assert(offsets[blockIdx.x+1]>=offsets[blockIdx.x]);
  if (size>0) radixSort(a,ind,size);

}

template<typename T>
__global__
void radixSortMultiWrapper(T * v, uint16_t * index, uint32_t * offsets) {
  radixSortMulti(v,index,offsets);
}

#endif // HeterogeneousCoreCUDAUtilities_radixSort_H
