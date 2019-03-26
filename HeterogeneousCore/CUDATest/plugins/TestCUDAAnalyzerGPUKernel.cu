#include "TestCUDAAnalyzerGPUKernel.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
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

TestCUDAAnalyzerGPUKernel::TestCUDAAnalyzerGPUKernel() {
  edm::Service<CUDAService> cs;
  auto streamPtr = cs->getCUDAStream();
  sum_ = cs->make_device_unique<float[]>(NUM_VALUES, *streamPtr);
  cudautils::memsetAsync(sum_, 0, NUM_VALUES, *streamPtr);
  // better to synchronize since there is no guarantee that the stream
  // of analyzeAsync() would be otherise synchronized with this one
  streamPtr->synchronize();
}

void TestCUDAAnalyzerGPUKernel::analyzeAsync(const float *d_input, cuda::stream_t<>& stream) const {
  analyze<<<int(ceil(float(NUM_VALUES)/256)), 256, 0, stream.id()>>>(d_input, sum_.get(), NUM_VALUES);
}

float TestCUDAAnalyzerGPUKernel::value() const {
  edm::Service<CUDAService> cs;
  auto streamPtr = cs->getCUDAStream();
  auto accumulator = cs->make_device_unique<float>(*streamPtr);
  auto h_accumulator = cs->make_host_unique<float>(*streamPtr);
  sum<<<1,1, 0, streamPtr->id()>>>(sum_.get(), accumulator.get(), NUM_VALUES);
  cudautils::copyAsync(h_accumulator, accumulator, *streamPtr);
  // need to synchronize
  streamPtr->synchronize();
  return *h_accumulator;
}
