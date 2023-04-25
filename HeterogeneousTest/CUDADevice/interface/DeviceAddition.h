#ifndef HeterogeneousTest_CUDADevice_interface_DeviceAddition_h
#define HeterogeneousTest_CUDADevice_interface_DeviceAddition_h

#include <cstddef>

#include <cuda_runtime.h>

namespace cms::cudatest {

  __device__ void add_vectors_f(const float* __restrict__ in1,
                                const float* __restrict__ in2,
                                float* __restrict__ out,
                                size_t size);

  __device__ void add_vectors_d(const double* __restrict__ in1,
                                const double* __restrict__ in2,
                                double* __restrict__ out,
                                size_t size);

}  // namespace cms::cudatest

#endif  // HeterogeneousTest_CUDADevice_interface_DeviceAddition_h
