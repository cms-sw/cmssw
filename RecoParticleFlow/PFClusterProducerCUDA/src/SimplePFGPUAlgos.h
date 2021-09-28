#ifndef RecoParticleFlow_PFClusterProducerCUDA_src_SimplePFGPUAlgos_h
#define RecoParticleFlow_PFClusterProducerCUDA_src_SimplePFGPUAlgos_h

#include "DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"

namespace hcal {
  namespace reconstruction {

    void entryPoint_for_PFComputation(hcal::reconstruction::OutputDataGPU const&,
				      hcal::reconstruction::OutputPFRecHitDataGPU&,
				      cudaStream_t);

}



}

#endif
