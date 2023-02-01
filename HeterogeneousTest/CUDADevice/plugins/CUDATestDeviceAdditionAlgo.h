#ifndef HeterogeneousTest_CUDADevice_plugins_CUDATestDeviceAdditionAlgo_h
#define HeterogeneousTest_CUDADevice_plugins_CUDATestDeviceAdditionAlgo_h

#include <cstddef>

namespace HeterogeneousCoreCUDATestDevicePlugins {

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size);

}  // namespace HeterogeneousCoreCUDATestDevicePlugins

#endif  // HeterogeneousTest_CUDADevice_plugins_CUDATestDeviceAdditionAlgo_h
