#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include <iostream>

template <memoryPool::Where where>
struct dataProducer {
  auto operator()(cudaStream_t stream) { return memoryPool::cuda::makeBuffer<int>(20, stream, where); }
};

int main() {
  {
    int devices = 0;
    auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || 0 == devices)
      return 0;
    std::cout << "found " << devices << " cuda devices" << std::endl;
  }
  const int NUMTHREADS = 1;

  printf("Using CUDA %d\n", CUDART_VERSION);
  int cuda_device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, cuda_device);
  printf("CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);

  cudaStream_t streams[NUMTHREADS];

  for (int i = 0; i < NUMTHREADS; i++) {
    cudaStreamCreate(&(streams[i]));
  }

  memoryPool::cuda::init(false);
  memoryPool::cuda::dumpStat();

  auto& stream = streams[0];

  {
    auto pd = memoryPool::cuda::makeBuffer<int>(20, stream, memoryPool::onDevice);
    auto ph = memoryPool::cuda::makeBuffer<int>(20, stream, memoryPool::onHost);
    auto pc = memoryPool::cuda::makeBuffer<int>(20, nullptr, memoryPool::onCPU);

    auto dp = dataProducer<memoryPool::onDevice>()(stream);

    cudaCheck(memoryPool::cuda::copy(ph, pd, 20, stream));
    cudaCheck(memoryPool::cuda::copy(pd, ph, 20, stream));
    std::cout << "expect 2a 2u 1a 1u" << std::endl;
    memoryPool::cuda::dumpStat();

    {
      auto ph = memoryPool::cuda::makeBuffer<int>(20, stream, memoryPool::onHost);
      cudaCheck(memoryPool::cuda::copy(pd, ph, 20, stream));
    }
    cudaStreamSynchronize(stream);
    std::cout << "expect 2a 2u 2a 1u " << std::endl;
    memoryPool::cuda::dumpStat();
    {
      auto ph = memoryPool::cuda::makeBuffer<int>(20, stream, memoryPool::onHost);
      cudaCheck(memoryPool::cuda::copy(pd, ph, 20, stream));
    }
    std::cout << "expect 2a 2u 2a 1u " << std::endl;
    cudaStreamSynchronize(stream);
    memoryPool::cuda::dumpStat();
  }
  std::cout << "expect 2a 0u 2a 0u " << std::endl;
  cudaStreamSynchronize(stream);
  memoryPool::cuda::dumpStat();

  {
    memoryPool::Deleter devDeleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream, memoryPool::onDevice));
    memoryPool::Deleter hosDeleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream, memoryPool::onHost));

    auto p0 = memoryPool::cuda::makeBuffer<int>(20, devDeleter);
    auto p1 = memoryPool::cuda::makeBuffer<double>(20, devDeleter);
    auto p2 = memoryPool::cuda::makeBuffer<bool>(20, devDeleter);
    auto p3 = memoryPool::cuda::makeBuffer<int>(20, devDeleter);

    {
      auto pd = memoryPool::cuda::makeBuffer<int>(40, stream, memoryPool::onDevice);
      p0.reset(pd.release());
      memoryPool::cuda::dumpStat();
    }
    cudaStreamSynchronize(stream);

    auto hp0 = memoryPool::cuda::makeBuffer<int>(40, hosDeleter);
    auto hp1 = memoryPool::cuda::makeBuffer<double>(20, hosDeleter);
    auto hp2 = memoryPool::cuda::makeBuffer<bool>(20, hosDeleter);
    auto hp3 = memoryPool::cuda::makeBuffer<int>(20, hosDeleter);

    cudaCheck(memoryPool::cuda::copy(hp3, p3, 20, stream));
    cudaCheck(memoryPool::cuda::copy(p0, hp0, 40, stream));
    ;

    memoryPool::cuda::dumpStat();
  }

  cudaStreamSynchronize(stream);
  memoryPool::cuda::dumpStat();
  memoryPool::cuda::shutdown();

  return 0;
}
