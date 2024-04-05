#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousTest/AlpakaDevice/interface/alpaka/DeviceAddition.h"
#include "HeterogeneousTest/AlpakaKernel/interface/alpaka/DeviceAdditionKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  ALPAKA_FN_ACC void KernelAddVectorsF::operator()(Acc1D const& acc,
                                                   const float* __restrict__ in1,
                                                   const float* __restrict__ in2,
                                                   float* __restrict__ out,
                                                   uint32_t size) const {
    add_vectors_f(acc, in1, in2, out, size);
  }

  ALPAKA_FN_ACC void KernelAddVectorsD::operator()(Acc1D const& acc,
                                                   const double* __restrict__ in1,
                                                   const double* __restrict__ in2,
                                                   double* __restrict__ out,
                                                   uint32_t size) const {
    add_vectors_d(acc, in1, in2, out, size);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test
