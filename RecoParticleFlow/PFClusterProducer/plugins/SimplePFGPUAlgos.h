#ifndef RecoParticleFlow_PFClusterProducer_plugins_SimplePFGPUAlgos_h
#define RecoParticleFlow_PFClusterProducer_plugins_SimplePFGPUAlgos_h

#include <array>

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"

#include "DeclsForKernels.h"

namespace PFRecHit {
  namespace HCAL {
    void initializeCudaConstants(const PFRecHit::HCAL::Constants& cudaConstants, const cudaStream_t cudaStream = cudaStreamDefault);

    void entryPoint(::hcal::RecHitCollection<::calo::common::DevStoragePolicy> const&,
                    OutputPFRecHitDataGPU&,
                    PersistentDataGPU&,
                    ScratchDataGPU&,
                    cudaStream_t,
                    std::array<float, 5>& timer);

  }  // namespace HCAL

  namespace ECAL {
    void initializeCudaConstants(const uint32_t in_nValidRHBarrel,
                                 const uint32_t in_nValidRHEndcap,
                                 const float in_qTestThresh);

    void entryPoint(::ecal::UncalibratedRecHit<::calo::common::DevStoragePolicy> const&,
                    OutputPFRecHitDataGPU&,
                    PersistentDataGPU&,
                    ScratchDataGPU&,
                    cudaStream_t);

  }  // namespace ECAL
}  // namespace PFRecHit

#endif  // RecoParticleFlow_PFClusterProducer_plugins_SimplePFGPUAlgos_h
