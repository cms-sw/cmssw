#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

struct CudaDeviceAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMalloc(&p, size);
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) { cudaFree(ptr); }
};

struct CudaHostAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMallocHost(&p, size);
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) { cudaFreeHost(ptr); }
};

namespace {

  constexpr int poolSize = 128 * 1024;

  SimplePoolAllocatorImpl<PosixAlloc> cpuPool(poolSize);

  SimplePoolAllocatorImpl<CudaHostAlloc> hostPool(poolSize);

  struct DevicePools {
    using Pool = SimplePoolAllocatorImpl<CudaDeviceAlloc>;
    DevicePools(int size) {
      int devices = 0;
      auto status = cudaGetDeviceCount(&devices);
      if (status == cudaSuccess && devices > 0) {
        m_devicePools.reserve(devices);
        for (int i = 0; i < devices; ++i)
          m_devicePools.emplace_back(new Pool(size));
      }
    }
    //return pool for current device
    Pool &operator()() {
      int dev = -1;
      cudaGetDevice(&dev);
      return *m_devicePools[dev];
    }

    std::vector<std::unique_ptr<Pool>> m_devicePools;
  };

  DevicePools devicePool(poolSize);

}  // namespace

namespace memoryPool {
  namespace cuda {

    void dumpStat() {
      std::cout << "device pool" << std::endl;
      devicePool().dumpStat();
      std::cout << "host pool" << std::endl;
      hostPool.dumpStat();
    }

    SimplePoolAllocator *getPool(Where where) {
      return onCPU == where
                 ? (SimplePoolAllocator *)(&cpuPool)
                 : (onDevice == where ? (SimplePoolAllocator *)(&devicePool()) : (SimplePoolAllocator *)(&hostPool));
    }

    struct Payload {
      SimplePoolAllocator *pool;
      std::vector<int> buckets;
    };

    //  free callback
    void CUDART_CB freeCallback(void *p) {
      auto payload = (Payload *)(p);
      auto &pool = *(payload->pool);
      auto const &buckets = payload->buckets;
      for (auto i : buckets) {
        pool.free(i);
      }
      delete payload;
    }

    // allocate either on current device or on host (actually anywhere, not cuda specific)
    std::pair<void *, int> alloc(uint64_t size, SimplePoolAllocator &pool) {
      int i = pool.alloc(size);
      void *p = pool.pointer(i);
      return std::pair<void *, int>(p, i);
    }

    // schedule free
    void free(cudaStream_t stream, std::vector<int> buckets, SimplePoolAllocator &pool) {
      auto payload = new Payload{&pool, std::move(buckets)};
      cudaLaunchHostFunc(stream, freeCallback, payload);
    }

  }  // namespace cuda
}  // namespace memoryPool
