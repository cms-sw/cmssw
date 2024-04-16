#ifndef HeterogeneousTest_AlpakaWrapper_interface_alpaka_DeviceAdditionWrapper_h
#define HeterogeneousTest_AlpakaWrapper_interface_alpaka_DeviceAdditionWrapper_h

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  void wrapper_add_vectors_f(Queue& queue,
                             const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             uint32_t size);

  void wrapper_add_vectors_d(Queue& queue,
                             const double* __restrict__ in1,
                             const double* __restrict__ in2,
                             double* __restrict__ out,
                             uint32_t size);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test

#endif  // HeterogeneousTest_AlpakaWrapper_interface_alpaka_DeviceAdditionWrapper_h
