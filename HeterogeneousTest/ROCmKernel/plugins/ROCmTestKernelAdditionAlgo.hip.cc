#include <cstddef>

#include <hip/hip_runtime.h>

#include "HeterogeneousTest/ROCmKernel/interface/DeviceAdditionKernel.h"
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"

#include "ROCmTestKernelAdditionAlgo.h"

namespace HeterogeneousCoreROCmTestKernelPlugins {

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size) {
    cms::rocmtest::kernel_add_vectors_f<<<32, 32>>>(in1, in2, out, size);
    hipCheck(hipGetLastError());
  }

}  // namespace HeterogeneousCoreROCmTestKernelPlugins
