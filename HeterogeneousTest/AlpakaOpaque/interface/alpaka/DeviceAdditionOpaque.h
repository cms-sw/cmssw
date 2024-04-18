#ifndef HeterogeneousTest_AlpakaOpaque_interface_alpaka_DeviceAdditionOpaque_h
#define HeterogeneousTest_AlpakaOpaque_interface_alpaka_DeviceAdditionOpaque_h

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  void opaque_add_vectors_f(const float* in1, const float* in2, float* out, uint32_t size);

  void opaque_add_vectors_d(const double* in1, const double* in2, double* out, uint32_t size);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test

#endif  // HeterogeneousTest_AlpakaOpaque_interface_alpaka_DeviceAdditionOpaque_h
