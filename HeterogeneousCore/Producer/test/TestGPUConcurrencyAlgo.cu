#include <cuda.h>
#include <cuda_runtime.h>

#include "TestGPUConcurrencyAlgo.h"

__global__
void kernel(uint32_t sleep) {
  volatile int sum = 0;
  auto index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < 32)
  for (uint32_t i = 0; i < sleep; ++i)
    sum += i;
}

void TestGPUConcurrencyAlgo::kernelWrapper(cudaStream_t stream) const {
  kernel<<<blocks_, threads_, 0, stream>>>(sleep_);
}
