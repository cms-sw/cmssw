#ifndef HeterogeneousTest_AlpakaDevice_interface_alpaka_DeviceAddition_h
#define HeterogeneousTest_AlpakaDevice_interface_alpaka_DeviceAddition_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace cms::alpakatest {

  template <typename TAcc>
  ALPAKA_FN_ACC void add_vectors_f(TAcc const& acc,
                                   float const* __restrict__ in1,
                                   float const* __restrict__ in2,
                                   float* __restrict__ out,
                                   uint32_t size) {
    for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
      out[i] = in1[i] + in2[i];
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC void add_vectors_d(TAcc const& acc,
                                   double const* __restrict__ in1,
                                   double const* __restrict__ in2,
                                   double* __restrict__ out,
                                   uint32_t size) {
    for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
      out[i] = in1[i] + in2[i];
    }
  }

}  // namespace cms::alpakatest

#endif  // HeterogeneousTest_AlpakaDevice_interface_alpaka_DeviceAddition_h
