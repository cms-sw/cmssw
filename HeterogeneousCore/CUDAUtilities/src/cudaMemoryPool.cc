#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include<iostream>

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

  }  // namespace cuda
}  // namespace memoryPool


