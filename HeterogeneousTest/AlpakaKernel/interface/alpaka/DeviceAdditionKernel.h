#ifndef HeterogeneousTest_AlpakaKernel_interface_alpaka_DeviceAdditionKernel_h
#define HeterogeneousTest_AlpakaKernel_interface_alpaka_DeviceAdditionKernel_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousTest/AlpakaDevice/interface/alpaka/DeviceAddition.h"

namespace cms::alpakatest {

  struct KernelAddVectorsF {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const float* __restrict__ in1,
                                  const float* __restrict__ in2,
                                  float* __restrict__ out,
                                  uint32_t size) const {
      add_vectors_f(acc, in1, in2, out, size);
    }
  };

  struct KernelAddVectorsD {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const double* __restrict__ in1,
                                  const double* __restrict__ in2,
                                  double* __restrict__ out,
                                  uint32_t size) const {
      add_vectors_d(acc, in1, in2, out, size);
    }
  };

}  // namespace cms::alpakatest

#endif  // HeterogeneousTest_AlpakaKernel_interface_alpaka_DeviceAdditionKernel_h
