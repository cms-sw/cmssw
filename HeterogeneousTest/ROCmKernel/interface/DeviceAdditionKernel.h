#ifndef HeterogeneousTest_ROCmKernel_interface_DeviceAdditionKernel_h
#define HeterogeneousTest_ROCmKernel_interface_DeviceAdditionKernel_h

#include <cstddef>

#include <hip/hip_runtime.h>

namespace cms::rocmtest {

  __global__ void kernel_add_vectors_f(const float* __restrict__ in1,
                                       const float* __restrict__ in2,
                                       float* __restrict__ out,
                                       size_t size);

  __global__ void kernel_add_vectors_d(const double* __restrict__ in1,
                                       const double* __restrict__ in2,
                                       double* __restrict__ out,
                                       size_t size);

}  // namespace cms::rocmtest

#endif  // HeterogeneousTest_ROCmKernel_interface_DeviceAdditionKernel_h
