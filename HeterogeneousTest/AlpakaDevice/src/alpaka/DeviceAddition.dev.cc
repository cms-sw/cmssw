#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousTest/AlpakaDevice/interface/alpaka/DeviceAddition.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  ALPAKA_FN_ACC void add_vectors_f(Acc1D const& acc,
                                   float const* __restrict__ in1,
                                   float const* __restrict__ in2,
                                   float* __restrict__ out,
                                   uint32_t size) {
    for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
      out[i] = in1[i] + in2[i];
    }
  }

  ALPAKA_FN_ACC void add_vectors_d(Acc1D const& acc,
                                   double const* __restrict__ in1,
                                   double const* __restrict__ in2,
                                   double* __restrict__ out,
                                   uint32_t size) {
    for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
      out[i] = in1[i] + in2[i];
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test
