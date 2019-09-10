#include "TestCUDAAnalyzerGPUKernel.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

namespace {
  __global__
  void analyze(const float *input, float *sum, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) {
      atomicAdd(sum+i, input[i]);
    }
  }

  __global__
  void sum(const float *input, float *output, int numElements) {
    float val = 0.f;
    for(int i=0; i<numElements; ++i) {
      val += input[i];
    }
    *output = val;
  }
}

TestCUDAAnalyzerGPUKernel::TestCUDAAnalyzerGPUKernel(cuda::stream_t<>& stream) {
  sum_ = cudautils::make_device_unique<float[]>(NUM_VALUES, stream);
  cudautils::memsetAsync(sum_, 0, NUM_VALUES, stream);
  // better to synchronize since there is no guarantee that the stream
  // of analyzeAsync() would be otherise synchronized with this one
  stream.synchronize();
}

void TestCUDAAnalyzerGPUKernel::analyzeAsync(const float *d_input, cuda::stream_t<>& stream) const {
  analyze<<<int(ceil(float(NUM_VALUES)/256)), 256, 0, stream.id()>>>(d_input, sum_.get(), NUM_VALUES);
}

float TestCUDAAnalyzerGPUKernel::value(cuda::stream_t<>& stream) const {
  auto accumulator = cudautils::make_device_unique<float>(stream);
  auto h_accumulator = cudautils::make_host_unique<float>(stream);
  sum<<<1,1, 0, stream.id()>>>(sum_.get(), accumulator.get(), NUM_VALUES);
  cudautils::copyAsync(h_accumulator, accumulator, stream);
  // need to synchronize
  stream.synchronize();
  return *h_accumulator;
}
