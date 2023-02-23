#ifndef RecoParticleFlow_PFClusterProducer_test_testKernels_h
#define RecoParticleFlow_PFClusterProducer_test_testKernels_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusteringParamsGPU.h"

namespace testPFlow {

  void testPFClusteringParamsEntryPoint(PFClusteringParamsGPU::DeviceProduct const& pfClusParams,
                                        cudaStream_t cudaStream);

}

#endif  // RecoParticleFlow_PFClusterProducer_test_testKernels_h
