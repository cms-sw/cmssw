#ifndef HeterogeneousTest_CUDAKernel_plugins_CUDATestKernelAdditionAlgo_h
#define HeterogeneousTest_CUDAKernel_plugins_CUDATestKernelAdditionAlgo_h

#include <cstddef>

namespace HeterogeneousCoreCUDATestKernelPlugins {

  void wrapper_add_vectors_f(const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             size_t size);

}  // namespace HeterogeneousCoreCUDATestKernelPlugins

#endif  // HeterogeneousTest_CUDAKernel_plugins_CUDATestKernelAdditionAlgo_h
