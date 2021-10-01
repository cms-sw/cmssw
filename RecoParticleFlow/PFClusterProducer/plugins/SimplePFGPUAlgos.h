#ifndef RecoParticleFlow_PFClusterProducerCUDA_plugins_SimplePFGPUAlgos_h
#define RecoParticleFlow_PFClusterProducerCUDA_plugins_SimplePFGPUAlgos_h

#include "RecoParticleFlow/PFClusterProducer/plugins/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"

namespace hcal {
  namespace reconstruction {

    //void entryPoint_for_PFComputation(hcal::reconstruction::OutputDataGPU const&,
    void entryPoint_for_PFComputation(::hcal::RecHitCollection<calo::common::DevStoragePolicy> const&,
				      OutputPFRecHitDataGPU&,
				      cudaStream_t);

  }
}

#endif
