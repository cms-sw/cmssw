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
        PFRecHits.pfrh_neighbours = cms::cuda::make_device_unique<int[]>(Num_rechits*8, cudaStream);
        PFRecHits.pfrh_neighbourInfos = cms::cuda::make_device_unique<short[]>(Num_rechits*8, cudaStream);


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
        cms::cuda::host::unique_ptr<int[]> rh_neighbours;
                
        void allocate(uint32_t length, cudaStream_t cudaStream) {
            rh_pos = cms::cuda::make_host_unique<float3[]>(sizeof(float3)*length, cudaStream);
            rh_detId = cms::cuda::make_host_unique<uint32_t[]>(sizeof(uint32_t)*length, cudaStream);
            rh_neighbours = cms::cuda::make_host_unique<int[]>(sizeof(int)*length*8, cudaStream);
        }
    };

    struct PersistentDataGPU {
        cms::cuda::device::unique_ptr<float3[]> rh_pos;
        cms::cuda::device::unique_ptr<uint32_t[]> rh_detId;
        cms::cuda::device::unique_ptr<int[]> rh_neighbours;

        void allocate(uint32_t length, cudaStream_t cudaStream) {
            rh_pos = cms::cuda::make_device_unique<float3[]>(sizeof(float3)*length, cudaStream);
            rh_detId = cms::cuda::make_device_unique<uint32_t[]>(sizeof(uint32_t)*length, cudaStream);
            rh_neighbours = cms::cuda::make_device_unique<int[]>(sizeof(int)*length*8, cudaStream);
        }
    };

    struct ScratchDataGPU {
        uint32_t maxSize;
        cms::cuda::device::unique_ptr<bool[]> rh_mask;
        cms::cuda::device::unique_ptr<int[]> rh_inputToFullIdx; // Used to build map from input rechit index to lookup table index
        cms::cuda::device::unique_ptr<int[]> rh_fullToInputIdx; // Used to build map from lookup table index to input rechit index
        cms::cuda::device::unique_ptr<int[]> pfrhToInputIdx;  // Map PFRecHit index to input rechit index (to account for rechits cut in quality tests)
        cms::cuda::device::unique_ptr<int[]> inputToPFRHIdx;  // Map input rechit index to PF rechit index

        void allocate(uint32_t length, cudaStream_t cudaStream) {
            maxSize = length;
            rh_mask = cms::cuda::make_device_unique<bool[]>(sizeof(bool)*length, cudaStream);
            rh_inputToFullIdx = cms::cuda::make_device_unique<int[]>(sizeof(int)*length, cudaStream);
            rh_fullToInputIdx = cms::cuda::make_device_unique<int[]>(sizeof(int)*length, cudaStream);
            pfrhToInputIdx = cms::cuda::make_device_unique<int[]>(sizeof(int)*length, cudaStream);
            inputToPFRHIdx = cms::cuda::make_device_unique<int[]>(sizeof(int)*length, cudaStream);
        }
    };
  } // namespace rechit 
} //  namespace pf 


#endif
