#ifndef RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernelsPhase2_h
#define RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernelsPhase2_h

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

namespace ecal {
  namespace weights {

    struct EventOutputDataGPU {
      UncalibratedRecHit<::calo::common::DevStoragePolicy> recHits;

      void allocate(uint32_t digi_size, cudaStream_t cudaStream) {
        auto const size = digi_size;
        recHits.amplitudesAll =
            cms::cuda::make_device_unique<reco::ComputationScalarType[]>(size * EcalDataFrame::MAXSAMPLES, cudaStream);
        recHits.amplitude = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        recHits.amplitudeError = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        recHits.chi2 = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        recHits.pedestal = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        recHits.did = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
        recHits.flags = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
      }
    };
  }  //namespace weights
}  //namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernelsPhase2_h
