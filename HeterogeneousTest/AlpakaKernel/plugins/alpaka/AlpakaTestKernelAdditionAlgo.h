#ifndef HeterogeneousTest_AlpakaKernel_plugins_alpaka_AlpakaTestKernelAdditionAlgo_h
#define HeterogeneousTest_AlpakaKernel_plugins_alpaka_AlpakaTestKernelAdditionAlgo_h

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::HeterogeneousTestAlpakaKernelPlugins {

  void wrapper_add_vectors_f(Queue& queue,
                             const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             uint32_t size);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::HeterogeneousTestAlpakaKernelPlugins

#endif  // HeterogeneousTest_AlpakaKernel_plugins_alpaka_AlpakaTestKernelAdditionAlgo_h
