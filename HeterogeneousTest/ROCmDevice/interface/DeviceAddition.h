#ifndef HeterogeneousTest_ROCmDevice_interface_DeviceAddition_h
#define HeterogeneousTest_ROCmDevice_interface_DeviceAddition_h

#include <cstddef>

#include <hip/hip_runtime.h>

namespace cms::rocmtest {

  __device__ void add_vectors_f(const float* __restrict__ in1,
                                const float* __restrict__ in2,
                                float* __restrict__ out,
                                size_t size);

  __device__ void add_vectors_d(const double* __restrict__ in1,
                                const double* __restrict__ in2,
                                double* __restrict__ out,
                                size_t size);

}  // namespace cms::rocmtest

#endif  // HeterogeneousTest_ROCmDevice_interface_DeviceAddition_h
