#ifndef RecoLocalCalo_HcalRecProducers_src_MahiGPU_h
#define RecoLocalCalo_HcalRecProducers_src_MahiGPU_h

#include "DeclsForKernels.h"
#include "KernelHelpers.h"

namespace hcal {
  namespace mahi {

    void entryPoint(InputDataGPU const&,
                    OutputDataGPU&,
                    ConditionsProducts const&,
                    ScratchDataGPU&,
                    ConfigParameters const&,
                    cudaStream_t);

  }
}  // namespace hcal

#endif  // RecoLocalCalo_HcalRecProducers_src_MahiGPU_h
