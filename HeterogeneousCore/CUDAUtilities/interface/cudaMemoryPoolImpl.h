// #include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>

namespace {

  //  free callback
  void CUDART_CB freeCallback(cudaStream_t streamId, cudaError_t status, void* p) {
  //void CUDART_CB freeCallback(void *p) {
    if (status != cudaSuccess) {
       std::cout << "Error in free callaback in stream " << streamId << std::endl;
       auto error = cudaGetErrorName(status);
       auto message = cudaGetErrorString(status);
       std::cout << " error " << error << ": " << message << std::endl;
    }
    // std::cout << "free callaback" << std::endl;
    auto payload = (memoryPool::Payload *)(p);
    memoryPool::scheduleFree(payload);
  }

}  // namespace

struct CudaAlloc {
  static void scheduleFree(memoryPool::Payload *payload, cudaStream_t stream) {
    // std::cout    << "schedule free" << std::endl;
    if (stream)
      cudaCheck(cudaStreamAddCallback(stream, freeCallback, payload,0));
      // cudaCheck(cudaLaunchHostFunc(stream, freeCallback, payload));
    else
      memoryPool::scheduleFree(payload);
  }
};

struct CudaDeviceAlloc : public CudaAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMalloc(&p, size);
    // std::cout << "alloc " << size << ((err == cudaSuccess) ? " ok" : " err") << std::endl;
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) { 
     auto err = cudaFree(ptr); 
     // std::cout << "free" << ((err == cudaSuccess) ? " ok" : " err") <<std::endl;
     if (err != cudaSuccess) std::cout << " error in cudaFree??" << std::endl;
  }
};

struct CudaHostAlloc : public CudaAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMallocHost(&p, size);
    // std::cout << "alloc H " << size << ((err == cudaSuccess) ? " ok" : " err") << std::endl;
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
      CudaHostAlloc::scheduleFree(payload, stream);
    }

  }  // namespace cuda
}  // namespace memoryPool
