#ifndef HeterogeneousCoreCUDAUtilities_radixSort_H
#define HeterogeneousCoreCUDAUtilities_radixSort_H

#ifdef __CUDACC__

#include <cstdint>
#include <type_traits>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

template <typename T>
__device__ inline void dummyReorder(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {}

template <typename T>
__device__ inline void reorderSigned(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
  //move negative first...

  int32_t first = threadIdx.x;
  __shared__ uint32_t firstNeg;
  firstNeg = a[ind[0]] < 0 ? 0 : size;
  __syncthreads();

  // find first negative
  for (auto i = first; i < size - 1; i += blockDim.x) {
    if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
      firstNeg = i + 1;
  }

  __syncthreads();

  auto ii = first;
  for (auto i = firstNeg + threadIdx.x; i < size; i += blockDim.x) {
    ind2[ii] = ind[i];
    ii += blockDim.x;
  }
  __syncthreads();
  ii = size - firstNeg + threadIdx.x;
  assert(ii >= 0);
  for (auto i = first; i < firstNeg; i += blockDim.x) {
    ind2[ii] = ind[i];
    ii += blockDim.x;
  }
  __syncthreads();
  for (auto i = first; i < size; i += blockDim.x)
    ind[i] = ind2[i];
}

template <typename T>
__device__ inline void reorderFloat(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
  //move negative first...

  int32_t first = threadIdx.x;
  __shared__ uint32_t firstNeg;
  firstNeg = a[ind[0]] < 0 ? 0 : size;
  __syncthreads();

  // find first negative
  for (auto i = first; i < size - 1; i += blockDim.x) {
    if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
      firstNeg = i + 1;
  }

  __syncthreads();

  int ii = size - firstNeg - threadIdx.x - 1;
  for (auto i = firstNeg + threadIdx.x; i < size; i += blockDim.x) {
    ind2[ii] = ind[i];
    ii -= blockDim.x;
  }
  __syncthreads();
  ii = size - firstNeg + threadIdx.x;
  assert(ii >= 0);
  for (auto i = first; i < firstNeg; i += blockDim.x) {
    ind2[ii] = ind[i];
    ii += blockDim.x;
  }
  __syncthreads();
  for (auto i = first; i < size; i += blockDim.x)
    ind[i] = ind2[i];
}

template <typename T,  // shall be interger
          int NS,      // number of significant bytes to use in sorting
          typename RF>
__device__ __forceinline__ void radixSortImpl(
    T const* __restrict__ a, uint16_t* ind, uint16_t* ind2, uint32_t size, RF reorder) {
  constexpr int d = 8, w = 8 * sizeof(T);
  constexpr int sb = 1 << d;
  constexpr int ps = int(sizeof(T)) - NS;

  __shared__ int32_t c[sb], ct[sb], cu[sb];

  __shared__ int ibs;
  __shared__ int p;

  assert(size > 0);
  assert(blockDim.x >= sb);

  // bool debug = false; // threadIdx.x==0 && blockIdx.x==5;

  p = ps;

  auto j = ind;
  auto k = ind2;

  int32_t first = threadIdx.x;
  for (auto i = first; i < size; i += blockDim.x)
    j[i] = i;
  __syncthreads();

  while (__syncthreads_and(p < w / d)) {
    if (threadIdx.x < sb)
      c[threadIdx.x] = 0;
    __syncthreads();

    // fill bins
    for (auto i = first; i < size; i += blockDim.x) {
      auto bin = (a[j[i]] >> d * p) & (sb - 1);
      atomicAdd(&c[bin], 1);
    }
    __syncthreads();

    // prefix scan "optimized"???...
    if (threadIdx.x < sb) {
      auto x = c[threadIdx.x];
      auto laneId = threadIdx.x & 0x1f;
      CMS_UNROLL_LOOP
      for (int offset = 1; offset < 32; offset <<= 1) {
        auto y = __shfl_up_sync(0xffffffff, x, offset);
        if (laneId >= offset)
          x += y;
      }
      ct[threadIdx.x] = x;
    }
    __syncthreads();
    if (threadIdx.x < sb) {
      auto ss = (threadIdx.x / 32) * 32 - 1;
      c[threadIdx.x] = ct[threadIdx.x];
      for (int i = ss; i > 0; i -= 32)
        c[threadIdx.x] += ct[i];
    }
    /* 
    //prefix scan for the nulls  (for documentation)
    if (threadIdx.x==0)
      for (int i = 1; i < sb; ++i) c[i] += c[i-1];
    */

    // broadcast
    ibs = size - 1;
    __syncthreads();
    while (__syncthreads_and(ibs > 0)) {
      int i = ibs - threadIdx.x;
      if (threadIdx.x < sb) {
        cu[threadIdx.x] = -1;
        ct[threadIdx.x] = -1;
      }
      __syncthreads();
      int32_t bin = -1;
      if (threadIdx.x < sb) {
        if (i >= 0) {
          bin = (a[j[i]] >> d * p) & (sb - 1);
          ct[threadIdx.x] = bin;
          atomicMax(&cu[bin], int(i));
        }
      }
      __syncthreads();
      if (threadIdx.x < sb) {
        if (i >= 0 && i == cu[bin])  // ensure to keep them in order
          for (int ii = threadIdx.x; ii < sb; ++ii)
            if (ct[ii] == bin) {
              auto oi = ii - threadIdx.x;
              // assert(i>=oi);if(i>=oi)
              k[--c[bin]] = j[i - oi];
            }
      }
      __syncthreads();
      if (bin >= 0)
        assert(c[bin] >= 0);
      if (threadIdx.x == 0)
        ibs -= sb;
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
    assert(c[0] == 0);

    // swap (local, ok)
    auto t = j;
    j = k;
    k = t;

    if (threadIdx.x == 0)
      ++p;
    __syncthreads();
  }

  if ((w != 8) && (0 == (NS & 1)))
    assert(j == ind);  // w/d is even so ind is correct

  if (j != ind)  // odd...
    for (auto i = first; i < size; i += blockDim.x)
      ind[i] = ind2[i];

  __syncthreads();

  // now move negative first... (if signed)
  reorder(a, ind, ind2, size);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_unsigned<T>::value, T>::type* = nullptr>
__device__ __forceinline__ void radixSort(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
  radixSortImpl<T, NS>(a, ind, ind2, size, dummyReorder<T>);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type* = nullptr>
__device__ __forceinline__ void radixSort(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
  radixSortImpl<T, NS>(a, ind, ind2, size, reorderSigned<T>);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
__device__ __forceinline__ void radixSort(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
  using I = int;
  radixSortImpl<I, NS>((I const*)(a), ind, ind2, size, reorderFloat<I>);
}

template <typename T, int NS = sizeof(T)>
__device__ __forceinline__ void radixSortMulti(T const* v,
                                               uint16_t* index,
                                               uint32_t const* offsets,
                                               uint16_t* workspace) {
  extern __shared__ uint16_t ws[];

  auto a = v + offsets[blockIdx.x];
  auto ind = index + offsets[blockIdx.x];
  auto ind2 = nullptr == workspace ? ws : workspace + offsets[blockIdx.x];
  auto size = offsets[blockIdx.x + 1] - offsets[blockIdx.x];
  assert(offsets[blockIdx.x + 1] >= offsets[blockIdx.x]);
  if (size > 0)
    radixSort<T, NS>(a, ind, ind2, size);
}

namespace cms {
  namespace cuda {

    template <typename T, int NS = sizeof(T)>
    __global__ void __launch_bounds__(256, 4)
        radixSortMultiWrapper(T const* v, uint16_t* index, uint32_t const* offsets, uint16_t* workspace) {
      radixSortMulti<T, NS>(v, index, offsets, workspace);
    }

    template <typename T, int NS = sizeof(T)>
    __global__ void radixSortMultiWrapper2(T const* v, uint16_t* index, uint32_t const* offsets, uint16_t* workspace) {
      radixSortMulti<T, NS>(v, index, offsets, workspace);
    }

  }  // namespace cuda
}  // namespace cms

#endif  // __CUDACC__

#endif  // HeterogeneousCoreCUDAUtilities_radixSort_H
