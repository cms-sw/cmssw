#ifndef HeterogeneousCore_CUDACore_TestCUDAProducerGPUKernel_h
#define HeterogeneousCore_CUDACore_TestCUDAProducerGPUKernel_h

#include <string>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

/**
 * This class models the actual CUDA implementation of an algorithm.
 *
 * Memory is allocated dynamically with the allocator in cms::cuda.
 *
 * The algorithm is intended to waste time with large matrix
 * operations so that the asynchronous nature of the CUDA integration
 * becomes visible with debug prints.
 */
class TestCUDAProducerGPUKernel {
public:
  static constexpr int NUM_VALUES = 4000;

  TestCUDAProducerGPUKernel() = default;
  ~TestCUDAProducerGPUKernel() = default;

  // returns (owning) pointer to device memory
  cms::cuda::device::unique_ptr<float[]> runAlgo(const std::string& label, cudaStream_t stream) const {
    return runAlgo(label, nullptr, stream);
  }
  cms::cuda::device::unique_ptr<float[]> runAlgo(const std::string& label,
                                                 const float* d_input,
                                                 cudaStream_t stream) const;

  void runSimpleAlgo(float* d_data, cudaStream_t stream) const;
};

#endif
