#include "TestCUDAAnalyzerGPUKernel.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace {
  __global__ void analyze(const float *input, float *sum, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
      atomicAdd(sum + i, input[i]);
    }
  }

  __global__ void sum(const float *input, float *output, int numElements) {
    float val = 0.f;
    for (int i = 0; i < numElements; ++i) {
      val += input[i];
    }
    *output = val;
  }
}  // namespace

TestCUDAAnalyzerGPUKernel::TestCUDAAnalyzerGPUKernel(cudaStream_t stream) {
  sum_ = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);
  cms::cuda::memsetAsync(sum_, 0, NUM_VALUES, stream);
  // better to synchronize since there is no guarantee that the stream
  // of analyzeAsync() would be otherwise synchronized with this one
  cudaCheck(cudaStreamSynchronize(stream));
}

void TestCUDAAnalyzerGPUKernel::analyzeAsync(const float *d_input, cudaStream_t stream) {
  analyze<<<int(ceil(float(NUM_VALUES) / 256)), 256, 0, stream>>>(d_input, sum_.get(), NUM_VALUES);
}

float TestCUDAAnalyzerGPUKernel::value(cudaStream_t stream) const {
  auto accumulator = cms::cuda::make_device_unique<float>(stream);
  auto h_accumulator = cms::cuda::make_host_unique<float>(stream);
  sum<<<1, 1, 0, stream>>>(sum_.get(), accumulator.get(), NUM_VALUES);
  cms::cuda::copyAsync(h_accumulator, accumulator, stream);
  // need to synchronize
  cudaCheck(cudaStreamSynchronize(stream));
  return *h_accumulator;
}
