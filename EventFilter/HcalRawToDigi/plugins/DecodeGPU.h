#ifndef EventFilter_HcalRawToDigi_interface_DecodeGPU_h
#define EventFilter_HcalRawToDigi_interface_DecodeGPU_h

#include "DeclsForKernels.h"

namespace hcal {
  namespace raw {

    void entryPoint(InputDataCPU const&,
                    InputDataGPU&,
                    OutputDataGPU&,
                    ScratchDataGPU&,
                    OutputDataCPU&,
                    ConditionsProducts const&,
                    ConfigurationParameters const&,
                    cudaStream_t cudaStream,
                    uint32_t const,
                    uint32_t const);

  }
}  // namespace hcal

#endif  // EventFilter_HcalRawToDigi_interface_DecodeGPU_h
