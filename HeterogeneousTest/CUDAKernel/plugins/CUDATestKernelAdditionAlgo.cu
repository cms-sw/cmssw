#include <cstddef>

#include <cuda_runtime.h>

#include "HeterogeneousTest/CUDAKernel/interface/DeviceAdditionKernel.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "CUDATestKernelAdditionAlgo.h"

namespace HeterogeneousCoreCUDATestKernelPlugins {

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size) {
    cms::cudatest::kernel_add_vectors_f<<<32, 32>>>(in1, in2, out, size);
    cudaCheck(cudaGetLastError());
  }

}  // namespace HeterogeneousCoreCUDATestKernelPlugins
