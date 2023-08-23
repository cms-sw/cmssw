#include <cstddef>

#include <hip/hip_runtime.h>

#include "HeterogeneousTest/ROCmDevice/interface/DeviceAddition.h"
#include "HeterogeneousTest/ROCmKernel/interface/DeviceAdditionKernel.h"

namespace cms::rocmtest {

  __global__ void kernel_add_vectors_f(const float* __restrict__ in1,
                                       const float* __restrict__ in2,
                                       float* __restrict__ out,
                                       size_t size) {
    add_vectors_f(in1, in2, out, size);
  }

  __global__ void kernel_add_vectors_d(const double* __restrict__ in1,
                                       const double* __restrict__ in2,
                                       double* __restrict__ out,
                                       size_t size) {
    add_vectors_d(in1, in2, out, size);
  }

}  // namespace cms::rocmtest
