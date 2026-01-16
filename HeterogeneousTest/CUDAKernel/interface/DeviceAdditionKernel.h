#ifndef HeterogeneousTest_CUDAKernel_interface_DeviceAdditionKernel_h
#define HeterogeneousTest_CUDAKernel_interface_DeviceAdditionKernel_h

#include <cstddef>

#include <cuda_runtime.h>

namespace cms::cudatest {

  // Mark the kernel with default visibility to export it as a public symbol for CUDA 12.8 and later
  __global__ __attribute__((visibility("default"))) void kernel_add_vectors_f(const float* __restrict__ in1,
                                                                              const float* __restrict__ in2,
                                                                              float* __restrict__ out,
                                                                              size_t size);

  // Mark the kernel with default visibility to export it as a public symbol for CUDA 12.8 and later
  __global__ __attribute__((visibility("default"))) void kernel_add_vectors_d(const double* __restrict__ in1,
                                                                              const double* __restrict__ in2,
                                                                              double* __restrict__ out,
                                                                              size_t size);

}  // namespace cms::cudatest

#endif  // HeterogeneousTest_CUDAKernel_interface_DeviceAdditionKernel_h
