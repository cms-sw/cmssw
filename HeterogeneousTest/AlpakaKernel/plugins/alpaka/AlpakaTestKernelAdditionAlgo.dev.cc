#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousTest/AlpakaKernel/interface/alpaka/DeviceAdditionKernel.h"

#include "AlpakaTestKernelAdditionAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::HeterogeneousTestAlpakaKernelPlugins {

  void wrapper_add_vectors_f(Queue& queue,
                             const float* __restrict__ in1,
                             const float* __restrict__ in2,
                             float* __restrict__ out,
                             uint32_t size) {
    alpaka::exec<Acc1D>(
        queue, cms::alpakatools::make_workdiv<Acc1D>(32, 32), cms::alpakatest::KernelAddVectorsF{}, in1, in2, out, size);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::HeterogeneousTestAlpakaKernelPlugins
