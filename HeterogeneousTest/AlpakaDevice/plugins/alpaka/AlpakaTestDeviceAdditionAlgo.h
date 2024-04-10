#ifndef HeterogeneousTest_AlpakaDevice_plugins_alpaka_AlpakaTestDeviceAdditionAlgo_h
#define HeterogeneousTest_AlpakaDevice_plugins_alpaka_AlpakaTestDeviceAdditionAlgo_h

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::HeterogeneousTestAlpakaDevicePlugins {

  void wrapper_add_vectors_f(Queue& queue,
                             const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             uint32_t size);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::HeterogeneousTestAlpakaDevicePlugins

#endif  // HeterogeneousTest_AlpakaDevice_plugins_alpaka_AlpakaTestDeviceAdditionAlgo_h
