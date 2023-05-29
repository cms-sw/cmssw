#include <cstddef>

#include <hip/hip_runtime.h>

#include "HeterogeneousTest/ROCmKernel/interface/DeviceAdditionKernel.h"
#include "HeterogeneousTest/ROCmWrapper/interface/DeviceAdditionWrapper.h"
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"

namespace cms::rocmtest {

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size) {
    // launch the 1-dimensional kernel for vector addition
    kernel_add_vectors_f<<<32, 32>>>(in1, in2, out, size);
    hipCheck(hipGetLastError());
  }

  void wrapper_add_vectors_d(const double* __restrict__ in1,
                             const double* __restrict__ in2,
                             double* __restrict__ out,
                             size_t size) {
    // launch the 1-dimensional kernel for vector addition
    kernel_add_vectors_d<<<32, 32>>>(in1, in2, out, size);
    hipCheck(hipGetLastError());
  }

}  // namespace cms::rocmtest
