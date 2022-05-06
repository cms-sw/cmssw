// #include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include<iostream>

namespace {

   //  free callback
    void CUDART_CB freeCallback(void *p) {
      // std::cout << "free callaback" << std::endl;
      auto payload = (memoryPool::Payload *)(p);
      memoryPool::scheduleFree(payload);
    }

}

struct CudaAlloc {
  static void  scheduleFree(memoryPool::Payload * payload, cudaStream_t stream) {
    // std::cout    << "schedule free" << std::endl;
    if (stream)
       cudaCheck(cudaLaunchHostFunc(stream, freeCallback, payload));
     else
       memoryPool::scheduleFree(payload);
  }
};


struct CudaDeviceAlloc : public CudaAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMalloc(&p, size);
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) { cudaFree(ptr); }

};

struct CudaHostAlloc : public CudaAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMallocHost(&p, size);
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) { cudaFreeHost(ptr); }
};


namespace memoryPool {
  namespace cuda {

    void dumpStat();

    SimplePoolAllocator *getPool(Where where);

    // allocate either on current device or on host (actually anywhere, not cuda specific)
    inline std::pair<void *, int> alloc(uint64_t size, SimplePoolAllocator &pool) {
      int i = pool.alloc(size);
      void *p = pool.pointer(i);
      return std::pair<void *, int>(p, i);
    }

    // schedule free
    inline void free(cudaStream_t stream, std::vector<int> buckets, SimplePoolAllocator &pool) {
      auto payload = new Payload{&pool, std::move(buckets)};
      CudaHostAlloc::scheduleFree(payload,stream);
    }

  }  // namespace cuda
}  // namespace memoryPool


