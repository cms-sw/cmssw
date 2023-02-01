#ifndef HeterogeneousTest_ROCmKernel_plugins_ROCmTestKernelAdditionAlgo_h
#define HeterogeneousTest_ROCmKernel_plugins_ROCmTestKernelAdditionAlgo_h

#include <cstddef>

namespace HeterogeneousCoreROCmTestKernelPlugins {

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size);

}  // namespace HeterogeneousCoreROCmTestKernelPlugins

#endif  // HeterogeneousTest_ROCmKernel_plugins_ROCmTestKernelAdditionAlgo_h
