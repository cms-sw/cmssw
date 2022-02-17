#ifndef RecoParticleFlow_PFClusterProducerCUDA_plugins_SimplePFGPUAlgos_h
#define RecoParticleFlow_PFClusterProducerCUDA_plugins_SimplePFGPUAlgos_h

#include "RecoParticleFlow/PFClusterProducer/plugins/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include <array>

namespace PFRecHit {
  namespace HCAL {
    void initializeCudaConstants(const uint32_t in_nValidRHBarrel,
                                 const uint32_t in_nValidRHEndcap,
                                 const float in_qTestThresh);

    void entryPoint(
                  ::hcal::RecHitCollection<::calo::common::DevStoragePolicy> const&,
                  OutputPFRecHitDataGPU&,
                  PersistentDataGPU&,
                  ScratchDataGPU&,
                  cudaStream_t,
                  std::array<float,5>& timer);

  }

  namespace ECAL {
    void initializeCudaConstants(const uint32_t in_nValidRHBarrel,
                                 const uint32_t in_nValidRHEndcap,
                                 const float in_qTestThresh);

    void entryPoint(
                  ::ecal::UncalibratedRecHit<::calo::common::DevStoragePolicy> const&,
                  OutputPFRecHitDataGPU&,
                  PersistentDataGPU&,
                  ScratchDataGPU&,
                  cudaStream_t);

  }
}

#endif
