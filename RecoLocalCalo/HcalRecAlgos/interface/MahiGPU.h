#ifndef RecoLocalCalo_HcalRecAlgos_interface_MahiGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_MahiGPU_h

#include "RecoLocalCalo/HcalRecAlgos/interface/DeclsForKernels.h"

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

#endif  // RecoLocalCalo_HcalRecAlgos_interface_MahiGPU_h
