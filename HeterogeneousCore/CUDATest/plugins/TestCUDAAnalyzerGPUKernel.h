#ifndef HeterogeneousCore_CUDACore_TestCUDAAnalyzerGPUKernel_h
#define HeterogeneousCore_CUDACore_TestCUDAAnalyzerGPUKernel_h

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include <cuda/api_wrappers.h>

class TestCUDAAnalyzerGPUKernel {
public:
  static constexpr int NUM_VALUES = 4000;

  TestCUDAAnalyzerGPUKernel();
  ~TestCUDAAnalyzerGPUKernel() = default;

  // returns (owning) pointer to device memory
  void analyzeAsync(const float *d_input, cuda::stream_t<>& stream) const;
  float value() const;

private:
  mutable cudautils::device::unique_ptr<float[]> sum_; // all writes are atomic in CUDA
};

#endif
