#ifndef HeterogeneousCore_CUDACore_TestCUDAAnalyzerGPUKernel_h
#define HeterogeneousCore_CUDACore_TestCUDAAnalyzerGPUKernel_h

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include <cuda_runtime.h>

class TestCUDAAnalyzerGPUKernel {
public:
  static constexpr int NUM_VALUES = 4000;

  TestCUDAAnalyzerGPUKernel(cudaStream_t stream);
  ~TestCUDAAnalyzerGPUKernel() = default;

  // thread safe
  void analyzeAsync(const float* d_input, cudaStream_t stream);
  float value(cudaStream_t stream) const;

private:
  cms::cuda::device::unique_ptr<float[]> sum_;  // all writes are atomic in CUDA
};

#endif
