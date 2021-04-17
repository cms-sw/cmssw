#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cstdint>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#ifdef __CUDA_ARCH__

template <typename T>
__device__ void __forceinline__ warpPrefixScan(T const* __restrict__ ci, T* __restrict__ co, uint32_t i, uint32_t mask) {
  // ci and co may be the same
  auto x = ci[i];
  auto laneId = threadIdx.x & 0x1f;
  CMS_UNROLL_LOOP
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
__device__ void __forceinline__ warpPrefixScan(T* c, uint32_t i, uint32_t mask) {
  auto x = c[i];
  auto laneId = threadIdx.x & 0x1f;
  CMS_UNROLL_LOOP
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

#endif

namespace cms {
  namespace cuda {

    // limited to 32*32 elements....
    template <typename VT, typename T>
    __host__ __device__ __forceinline__ void blockPrefixScan(VT const* ci,
                                                             VT* co,
                                                             uint32_t size,
                                                             T* ws
#ifndef __CUDA_ARCH__
                                                             = nullptr
#endif
    ) {
#ifdef __CUDA_ARCH__
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDim.x % 32);
      auto first = threadIdx.x;
      auto mask = __ballot_sync(0xffffffff, first < size);

      for (auto i = first; i < size; i += blockDim.x) {
        warpPrefixScan(ci, co, i, mask);
        auto laneId = threadIdx.x & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = co[i];
        mask = __ballot_sync(mask, i + blockDim.x < size);
      }
      __syncthreads();
      if (size <= 32)
        return;
      if (threadIdx.x < 32)
        warpPrefixScan(ws, threadIdx.x, 0xffffffff);
      __syncthreads();
      for (auto i = first + 32; i < size; i += blockDim.x) {
        auto warpId = i / 32;
        co[i] += ws[warpId - 1];
      }
      __syncthreads();
#else
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
#endif
    }

    // same as above, may remove
    // limited to 32*32 elements....
    template <typename T>
    __host__ __device__ __forceinline__ void blockPrefixScan(T* c,
                                                             uint32_t size,
                                                             T* ws
#ifndef __CUDA_ARCH__
                                                             = nullptr
#endif
    ) {
#ifdef __CUDA_ARCH__
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDim.x % 32);
      auto first = threadIdx.x;
      auto mask = __ballot_sync(0xffffffff, first < size);

      for (auto i = first; i < size; i += blockDim.x) {
        warpPrefixScan(c, i, mask);
        auto laneId = threadIdx.x & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = c[i];
        mask = __ballot_sync(mask, i + blockDim.x < size);
      }
      __syncthreads();
      if (size <= 32)
        return;
      if (threadIdx.x < 32)
        warpPrefixScan(ws, threadIdx.x, 0xffffffff);
      __syncthreads();
      for (auto i = first + 32; i < size; i += blockDim.x) {
        auto warpId = i / 32;
        c[i] += ws[warpId - 1];
      }
      __syncthreads();
#else
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
#endif
    }

#ifdef __CUDA_ARCH__
    // see https://stackoverflow.com/questions/40021086/can-i-obtain-the-amount-of-allocated-dynamic-shared-memory-from-within-a-kernel/40021087#40021087
    __device__ __forceinline__ unsigned dynamic_smem_size() {
      unsigned ret;
      asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
      return ret;
    }
#endif

    // in principle not limited....
    template <typename T>
    __global__ void multiBlockPrefixScan(T const* ici, T* ico, int32_t size, int32_t* pc) {
      volatile T const* ci = ici;
      volatile T* co = ico;
      __shared__ T ws[32];
#ifdef __CUDA_ARCH__
      assert(sizeof(T) * gridDim.x <= dynamic_smem_size());  // size of psum below
#endif
      assert(blockDim.x * gridDim.x >= size);
      // first each block does a scan
      int off = blockDim.x * blockIdx.x;
      if (size - off > 0)
        blockPrefixScan(ci + off, co + off, std::min(int(blockDim.x), size - off), ws);

      // count blocks that finished
      __shared__ bool isLastBlockDone;
      if (0 == threadIdx.x) {
        __threadfence();
        auto value = atomicAdd(pc, 1);  // block counter
        isLastBlockDone = (value == (int(gridDim.x) - 1));
      }

      __syncthreads();

      if (!isLastBlockDone)
        return;

      assert(int(gridDim.x) == *pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      extern __shared__ T psum[];
      for (int i = threadIdx.x, ni = gridDim.x; i < ni; i += blockDim.x) {
        auto j = blockDim.x * i + blockDim.x - 1;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      __syncthreads();
      blockPrefixScan(psum, psum, gridDim.x, ws);

      // now it would have been handy to have the other blocks around...
      for (int i = threadIdx.x + blockDim.x, k = 0; i < size; i += blockDim.x, ++k) {
        co[i] += psum[k];
      }
    }
  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
