#ifndef RecoParticleFlow_PFClusterProducerCUDA_plugins_SimplePFGPUAlgos_h
#define RecoParticleFlow_PFClusterProducerCUDA_plugins_SimplePFGPUAlgos_h

#include "RecoParticleFlow/PFClusterProducer/plugins/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"

namespace pf {
  namespace rechit {
    void initializeCudaConstants(const uint32_t in_nValidRHBarrel,
                                 const uint32_t in_nValidRHEndcap,
                                 const float in_qTestThresh);

    void entryPoint(
                  ::hcal::RecHitCollection<::calo::common::DevStoragePolicy> const&,
                  OutputPFRecHitDataGPU&,
                  PersistentDataGPU&,
                  cudaStream_t);

  }
}

#endif
