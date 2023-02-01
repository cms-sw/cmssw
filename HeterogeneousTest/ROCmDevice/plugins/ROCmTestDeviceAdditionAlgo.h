#ifndef HeterogeneousTest_ROCmDevice_plugins_ROCmTestDeviceAdditionAlgo_h
#define HeterogeneousTest_ROCmDevice_plugins_ROCmTestDeviceAdditionAlgo_h

#include <cstddef>

namespace HeterogeneousCoreROCmTestDevicePlugins {

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size);

}  // namespace HeterogeneousCoreROCmTestDevicePlugins

#endif  // HeterogeneousTest_ROCmDevice_plugins_ROCmTestDeviceAdditionAlgo_h
