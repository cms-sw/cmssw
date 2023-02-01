#ifndef HeterogeneousTest_CUDAOpaque_interface_DeviceAdditionOpaque_h
#define HeterogeneousTest_CUDAOpaque_interface_DeviceAdditionOpaque_h

#include <cstddef>

namespace cms::cudatest {

  void opaque_add_vectors_f(const float* in1, const float* in2, float* out, size_t size);

  void opaque_add_vectors_d(const double* in1, const double* in2, double* out, size_t size);

}  // namespace cms::cudatest

#endif  // HeterogeneousTest_CUDAOpaque_interface_DeviceAdditionOpaque_h
