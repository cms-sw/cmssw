#ifndef EventFilter_EcalRawToDigi_plugins_UnpackGPU_h
#define EventFilter_EcalRawToDigi_plugins_UnpackGPU_h

#include "DeclsForKernels.h"

namespace ecal {
  namespace raw {

    // FIXME: bundle up uint32_t values
    void entryPoint(InputDataCPU const&,
                    InputDataGPU&,
                    OutputDataGPU&,
                    ScratchDataGPU&,
                    OutputDataCPU&,
                    ConditionsProducts const&,
                    cudaStream_t,
                    uint32_t const,
                    uint32_t const);

  }  // namespace raw
}  // namespace ecal

#endif  // EventFilter_EcalRawToDigi_plugins_UnpackGPU_h
