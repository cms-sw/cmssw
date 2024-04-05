#ifndef HeterogeneousTest_AlpakaKernel_interface_alpaka_DeviceAdditionKernel_h
#define HeterogeneousTest_AlpakaKernel_interface_alpaka_DeviceAdditionKernel_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  struct KernelAddVectorsF {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const float* __restrict__ in1,
                                  const float* __restrict__ in2,
                                  float* __restrict__ out,
                                  uint32_t size) const;
  };

  struct KernelAddVectorsD {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const double* __restrict__ in1,
                                  const double* __restrict__ in2,
                                  double* __restrict__ out,
                                  uint32_t size) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test

#endif  // HeterogeneousTest_AlpakaKernel_interface_alpaka_DeviceAdditionKernel_h
