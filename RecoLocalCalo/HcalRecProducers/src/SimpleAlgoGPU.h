#ifndef RecoLocalCalo_HcalRecProducers_src_SimpleAlgoGPU_h
#define RecoLocalCalo_HcalRecProducers_src_SimpleAlgoGPU_h

#include "DeclsForKernels.h"

namespace hcal {
  namespace reconstruction {

    void entryPoint(InputDataGPU const&,
                    OutputDataGPU&,
                    ConditionsProducts const&,
                    ScratchDataGPU&,
                    ConfigParameters const&,
                    cudaStream_t);

  }
}  // namespace hcal

#endif  // RecoLocalCalo_HcalRecProducers_src_SimpleAlgoGPU_h
