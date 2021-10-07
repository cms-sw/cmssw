#ifndef RecoParticleFlow_PFClusterProducerCUDA_src_DeclsForKernels_h
#define RecoParticleFlow_PFClusterProducerCUDA_src_DeclsForKernels_h

#include <functional>
#include <optional>


#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace pf {
  namespace rechit {
    
    struct OutputPFRecHitDataGPU {
      ::hcal::PFRecHitCollection<::calo::common::DevStoragePolicy> PFRecHits;

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

  

    struct PersistentDataCPU {
        cms::cuda::host::unique_ptr<float3[]> rh_pos;
        cms::cuda::host::unique_ptr<uint32_t[]> rh_detId;
                
        void allocate(uint32_t length, cudaStream_t cudaStream) {
            rh_pos = cms::cuda::make_host_unique<float3[]>(sizeof(float3)*length, cudaStream);
            rh_detId = cms::cuda::make_host_unique<uint32_t[]>(sizeof(uint32_t)*length, cudaStream);
        }
    };

    struct PersistentDataGPU {
        cms::cuda::device::unique_ptr<float3[]> rh_pos;
        cms::cuda::device::unique_ptr<uint32_t[]> rh_detId;
        cms::cuda::device::unique_ptr<uint32_t[]> rh_detIdMap; // Used to build map from rechit detId to lookup table index

        void allocate(uint32_t length, cudaStream_t cudaStream) {
            rh_pos = cms::cuda::make_device_unique<float3[]>(sizeof(float3)*length, cudaStream);
            rh_detId = cms::cuda::make_device_unique<uint32_t[]>(sizeof(uint32_t)*length, cudaStream);
            rh_detIdMap = cms::cuda::make_device_unique<uint32_t[]>(sizeof(uint32_t)*length, cudaStream);
        }
    };
  } // namespace rechit 
} //  namespace pf 


#endif
