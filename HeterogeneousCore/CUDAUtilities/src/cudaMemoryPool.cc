#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>

namespace {

  //  free callback
  /*
  void CUDART_CB freeCallback(cudaStream_t streamId, cudaError_t status, void *p) {
    if (status != cudaSuccess) {
      std::cout << "Error in free callaback in stream " << streamId << std::endl;
      auto error = cudaGetErrorName(status);
      auto message = cudaGetErrorString(status);
      std::cout << " error " << error << ": " << message << std::endl;
    }
  */
  void CUDART_CB freeCallback(void *p) {
    // std::cout << "free callaback for stream " << streamId << std::endl;
    auto payload = (memoryPool::Payload *)(p);
    poolDetails::freeAsync(payload);
  }

}  // namespace

struct CudaAlloc {
  static void scheduleFree(memoryPool::Payload *payload, void *stream) {
    //if (cudaStreamQuery((cudaStream_t)(stream)) == cudaSuccess)
    //  poolDetails::freeAsync(payload);
    //else
    //cudaCheck(cudaStreamAddCallback((cudaStream_t)(stream), freeCallback, payload, 0));
    cudaCheck(cudaLaunchHostFunc((cudaStream_t)(stream), freeCallback, payload));
  }
};

struct CudaDeviceAlloc : public CudaAlloc {
  using Pointer = void *;

  static constexpr bool useScheduled = true;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMalloc(&p, size);
    // std::cout << "alloc " << size << ((err == cudaSuccess) ? " ok" : " err") << std::endl;
    auto err2 = cudaGetLastError();
    assert(err == err2);
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) {
    if (ptr == nullptr)
      std::cout << "free nullptr???" << std::endl;
    auto err = cudaFree(ptr);
    // std::cout << "free" << ((err == cudaSuccess) ? " ok" : " err") <<std::endl;
    if (err != cudaSuccess)
      std::cout << " error in cudaFree??" << std::endl;
  }
};

struct CudaHostAlloc : public CudaAlloc {
  using Pointer = void *;

  static constexpr bool useScheduled = false;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMallocHost(&p, size);
    // std::cout << "alloc H " << size << ((err == cudaSuccess) ? " ok" : " err") << std::endl;
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) { cudaFreeHost(ptr); }
};

namespace {

  // FIXME : move it in its own place
  std::unique_ptr<SimplePoolAllocatorImpl<PosixAlloc>> cpuPool;

  std::unique_ptr<SimplePoolAllocatorImpl<CudaHostAlloc>> hostPool;

  using DevicePool = SimplePoolAllocatorImpl<CudaDeviceAlloc>;
  std::vector<std::unique_ptr<DevicePool>> devicePools;

  void initDevicePools(int size) {
    int devices = 0;
    auto status = cudaGetDeviceCount(&devices);
    if (status == cudaSuccess && devices > 0) {
      devicePools.reserve(devices);
      for (int i = 0; i < devices; ++i)
        devicePools.emplace_back(new DevicePool(size));
    }
  }

  DevicePool *getDevicePool() {
    int dev = -1;
    cudaGetDevice(&dev);
    return devicePools[dev].get();
  }

}  // namespace

namespace memoryPool {
  namespace cuda {

    void init(bool onlyCPU) {
      constexpr int poolSize = 32 * 1024;
      cpuPool = std::make_unique<SimplePoolAllocatorImpl<PosixAlloc>>(poolSize);
      if (onlyCPU)
        return;
      initDevicePools(poolSize);
      hostPool = std::make_unique<SimplePoolAllocatorImpl<CudaHostAlloc>>(poolSize);
    }

    void shutdown() {
      cpuPool.reset();
      devicePools.clear();
      hostPool.reset();
    }

    void dumpStat() {
      std::cout << "device pool" << std::endl;
      getDevicePool()->dumpStat();
      std::cout << "host pool" << std::endl;
      hostPool->dumpStat();
    }

    SimplePoolAllocator *getPool(Where where) {
      return onCPU == where ? (SimplePoolAllocator *)(cpuPool.get())
                            : (onDevice == where ? (SimplePoolAllocator *)(getDevicePool())
                                                 : (SimplePoolAllocator *)(hostPool.get()));
    }

    // allocate either on current device or on host (actually anywhere, not cuda specific)
    std::tuple<void *, int, uint64_t> alloc(void *stream, uint64_t size, SimplePoolAllocator &pool) {
      int i = pool.alloc(size, stream);
      void *p = pool.pointer(i);
      uint64_t c = pool.count(i);
      return std::tuple<void *, int, uint64_t>(p, i, c);
    }

    // schedule free
    void free(void *stream, std::vector<std::pair<int, uint64_t>> buckets, SimplePoolAllocator &pool) {
      auto payload = new Payload{&pool, std::move(buckets)};
      pool.scheduleFree(payload, stream);
    }

  }  // namespace cuda
}  // namespace memoryPool
