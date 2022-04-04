#ifndef RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernels_h

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/RecoTypes.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatiosGPU.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRefGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphasGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrectionsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalMultifitParametersGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalPedestalsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariancesGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapesGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRechitADCToGeVConstantGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRechitChannelStatusGPU.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelationGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrectionsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstantsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

#include "EigenMatrixTypes_gpu.h"

struct EcalPulseShape;
class EcalSampleMask;
class EcalTimeBiasCorrections;
struct EcalPulseCovariance;
class EcalDigiCollection;
class EcalXtalGroupId;
class EcalSamplesCorrelation;
class EBDigiCollection;
class EEDigiCollection;

namespace ecal {
  namespace weights {

    struct EventOutputDataGPUWeights {
      UncalibratedRecHit<::calo::common::DevStoragePolicy> recHitsEB;

      void allocate(uint32_t digi_size, cudaStream_t cudaStream) {
        auto const sizeEB = digi_size;
        recHitsEB.amplitudesAll = cms::cuda::make_device_unique<reco::ComputationScalarType[]>(
            sizeEB * EcalDataFrame_Ph2::MAXSAMPLES, cudaStream);
        recHitsEB.amplitude = cms::cuda::make_device_unique<reco::StorageScalarType[]>(sizeEB, cudaStream);
        recHitsEB.amplitudeError = cms::cuda::make_device_unique<reco::StorageScalarType[]>(sizeEB, cudaStream);
        recHitsEB.chi2 = cms::cuda::make_device_unique<reco::StorageScalarType[]>(sizeEB, cudaStream);
        recHitsEB.pedestal = cms::cuda::make_device_unique<reco::StorageScalarType[]>(sizeEB, cudaStream);
        recHitsEB.did = cms::cuda::make_device_unique<uint32_t[]>(sizeEB, cudaStream);
        recHitsEB.flags = cms::cuda::make_device_unique<uint32_t[]>(sizeEB, cudaStream);
      }
    };
  }  //namespace weights
}  //namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernels_h
