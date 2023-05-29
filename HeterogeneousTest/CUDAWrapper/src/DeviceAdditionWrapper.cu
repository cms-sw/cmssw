#include <cstddef>

#include <cuda_runtime.h>

#include "HeterogeneousTest/CUDAKernel/interface/DeviceAdditionKernel.h"
#include "HeterogeneousTest/CUDAWrapper/interface/DeviceAdditionWrapper.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace cms::cudatest {

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size) {
    // launch the 1-dimensional kernel for vector addition
    kernel_add_vectors_f<<<32, 32>>>(in1, in2, out, size);
    cudaCheck(cudaGetLastError());
  }

  void wrapper_add_vectors_d(const double* __restrict__ in1,
                             const double* __restrict__ in2,
                             double* __restrict__ out,
                             size_t size) {
    // launch the 1-dimensional kernel for vector addition
    kernel_add_vectors_d<<<32, 32>>>(in1, in2, out, size);
    cudaCheck(cudaGetLastError());
  }

}  // namespace cms::cudatest
