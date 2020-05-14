#ifndef RecoLocalCalo_EcalRecAlgos_src_AmplitudeComputationKernels
#define RecoLocalCalo_EcalRecAlgos_src_AmplitudeComputationKernels

#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes_gpu.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/DeclsForKernels.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/Common.h"

class EcalPulseShape;
class EcalPulseCovariance;
class EcalUncalibratedRecHit;

namespace ecal {
  namespace multifit {

    namespace v1 {

      void minimization_procedure(EventInputDataGPU const& eventInputGPU,
                                  EventOutputDataGPU& eventOutputGPU,
                                  EventDataForScratchGPU& scratch,
                                  ConditionsProducts const& conditions,
                                  ConfigurationParameters const& configParameters,
                                  cudaStream_t cudaStream);

    }

  }  // namespace multifit
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecAlgos_src_AmplitudeComputationKernelsV1
