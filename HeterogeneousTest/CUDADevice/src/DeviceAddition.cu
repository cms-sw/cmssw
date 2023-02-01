#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "HeterogeneousTest/CUDADevice/interface/DeviceAddition.h"

namespace cms::cudatest {

  __device__ void add_vectors_f(const float* __restrict__ in1,
                                const float* __restrict__ in2,
                                float* __restrict__ out,
                                size_t size) {
    uint32_t thread = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (size_t i = thread; i < size; i += stride) {
      out[i] = in1[i] + in2[i];
    }
  }

  __device__ void add_vectors_d(const double* __restrict__ in1,
                                const double* __restrict__ in2,
                                double* __restrict__ out,
                                size_t size) {
    uint32_t thread = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (size_t i = thread; i < size; i += stride) {
      out[i] = in1[i] + in2[i];
    }
  }

}  // namespace cms::cudatest
