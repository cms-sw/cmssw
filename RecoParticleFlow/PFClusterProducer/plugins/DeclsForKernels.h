#ifndef RecoParticleFlow_PFClusterProducerCUDA_src_DeclsForKernels_h
#define RecoParticleFlow_PFClusterProducerCUDA_src_DeclsForKernels_h

#include <functional>
#include <optional>


#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace hcal {
  namespace reconstruction {
    
    struct OutputPFRecHitDataGPU {
      PFRecHitCollection<::calo::common::DevStoragePolicy> PFRecHits;

      void allocate(size_t Num_rechits, cudaStream_t cudaStream) {
        PFRecHits.pfrh_depth = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_layer = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_caloId = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_detId = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);


        PFRecHits.pfrh_time = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_energy = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_pt2 = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_x = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_y = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_z = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
      }
    };

  
  } // namespace reconstruction
} //  namespace hcal


#endif
