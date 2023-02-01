#include <cstddef>

#include <hip/hip_runtime.h>

#include "HeterogeneousTest/ROCmDevice/interface/DeviceAddition.h"
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"

#include "ROCmTestDeviceAdditionAlgo.h"

namespace HeterogeneousCoreROCmTestDevicePlugins {

  __global__ void kernel_add_vectors_f(const float* __restrict__ in1,
                                       const float* __restrict__ in2,
                                       float* __restrict__ out,
                                       size_t size) {
    cms::rocmtest::add_vectors_f(in1, in2, out, size);
  }

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size) {
    kernel_add_vectors_f<<<32, 32>>>(in1, in2, out, size);
    hipCheck(hipGetLastError());
  }

}  // namespace HeterogeneousCoreROCmTestDevicePlugins
