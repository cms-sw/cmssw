#include <cstddef>

#include <cuda_runtime.h>

#include "HeterogeneousTest/CUDADevice/interface/DeviceAddition.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "CUDATestDeviceAdditionAlgo.h"

namespace HeterogeneousCoreCUDATestDevicePlugins {

  __global__ void kernel_add_vectors_f(const float* __restrict__ in1,
                                       const float* __restrict__ in2,
                                       float* __restrict__ out,
                                       size_t size) {
    cms::cudatest::add_vectors_f(in1, in2, out, size);
  }

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size) {
    kernel_add_vectors_f<<<32, 32>>>(in1, in2, out, size);
    cudaCheck(cudaGetLastError());
  }

}  // namespace HeterogeneousCoreCUDATestDevicePlugins
