// -*- c++ -*-
#include <Eigen/Dense>

#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/SimplePFGPUAlgos.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

// Uncomment for debug mode
#define DEBUG_ENABLE


namespace pf {
  namespace rechit {
    
    __constant__ uint32_t nValidRHBarrel;
    __constant__ uint32_t nValidRHEndcap;
    __constant__ uint32_t nValidRHTotal;
    __constant__ float  qTestThresh;
   
    void initializeCudaConstants(const uint32_t in_nValidRHBarrel,
                                 const uint32_t in_nValidRHEndcap,
                                 const float in_qTestThresh) {

        cudaCheck(cudaMemcpyToSymbolAsync(nValidRHBarrel, &in_nValidRHBarrel, sizeof(uint32_t)));
#ifdef DEBUG_ENABLE
        printf("--- HCAL Cuda constant values ---\n");
        uint32_t ival = 0;
        cudaCheck(cudaMemcpyFromSymbol(&ival, nValidRHBarrel, sizeof(uint32_t)));
        printf("nValidRHBarrel read from symbol: %u\n", ival);
#endif
        
        cudaCheck(cudaMemcpyToSymbolAsync(nValidRHEndcap, &in_nValidRHEndcap, sizeof(uint32_t)));
#ifdef DEBUG_ENABLE
        ival = 0;
        cudaCheck(cudaMemcpyFromSymbol(&ival, nValidRHEndcap, sizeof(uint32_t)));
        printf("nValidRHEndcap read from symbol: %u\n", ival);
#endif
    
        uint32_t total = in_nValidRHBarrel + in_nValidRHEndcap;
        cudaCheck(cudaMemcpyToSymbolAsync(nValidRHTotal, &total, sizeof(uint32_t)));
#ifdef DEBUG_ENABLE
        ival = 0;
        cudaCheck(cudaMemcpyFromSymbol(&ival, nValidRHTotal, sizeof(uint32_t)));
        printf("nValidRHTotal read from symbol: %u\n", ival);
#endif
    
        cudaCheck(cudaMemcpyToSymbolAsync(qTestThresh, &in_qTestThresh, sizeof(float)));
#ifdef DEBUG_ENABLE
        float val = 0;
        cudaCheck(cudaMemcpyFromSymbol(&val, qTestThresh, sizeof(float)));
        printf("qTestThresh read from symbol: %f\n\n", val);
#endif
    }

    __global__ void convert_rechits_to_PFRechits(uint32_t size,
                         const float3* rh_pos,
                         const uint32_t* rh_neighbours,
                         const uint32_t* rh_detIdMap,
                         float const* recHits_energy,
						 float const* recHits_chi2,
						 float const* recHits_energyM0,
						 float const* recHits_timeM0,
						 uint32_t const* recHits_did,
						 int* pfrechits_depth,
						 int* pfrechits_layer,
						 int* pfrechits_caloId,
						 int* pfrechits_detId,
						 float* pfrechits_time,
						 float* pfrechits_energy,
						 float* pfrechits_pt2,
						 float* pfrechits_x,
						 float* pfrechits_y,
						 float* pfrechits_z) {

      
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x*gridDim.x) {

          pfrechits_time[i] = recHits_timeM0[i];
          pfrechits_energy[i] = recHits_energyM0[i];
          pfrechits_detId[i] = recHits_did[i];

          uint32_t index = rh_detIdMap[i];  // Determine table index corresponding to this detId
          float3 pos = rh_pos[index];   // position vector of this rechit
          pfrechits_x[i] = pos.x; 
          pfrechits_y[i] = pos.y; 
          pfrechits_z[i] = pos.z;
          
          
#ifdef DEBUG_ENABLE
          if (pfrechits_detId[i] == 1158694936) {
            printf("Cuda kernel found neighbours of 1158694936:\n\n");
            for (int i = 0; i < 8; i++)
                printf("\t%u\n", rh_neighbours[index*8+i]);
          }
#endif
      }

#ifdef DEBUG_ENABLE
//      if ((blockIdx.x * blockDim.x + threadIdx.x) < 5) {
//        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//        printf("Rechit %u with detId %u has position: (%f, %f, %f)\n", i, pfrechits_detId[i], pfrechits_x[i], pfrechits_y[i], pfrechits_z[i]);
//      }
#endif
    }

  
  __global__ void buildDetIdMapPerBlockMulti(uint32_t size,
                                uint32_t const* rh_detIdRef, // Lookup table index -> detId
                                uint32_t* rh_detIdMap,  //  Map for input rh detId -> lookup table index
                                uint32_t const* recHits_did) {  //  Rechit detIds
    
    __shared__ uint32_t detId, subdet, minval, maxval, notDone;

    for (uint32_t i = blockIdx.x; i < size; i += gridDim.x) {
        if (threadIdx.x == 0) {
            notDone = 1;
            detId = recHits_did[i];
            
            // Get subdetector encoded in detId
            // cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0048
            subdet = (detId >> 25) & 0x7;
            if (subdet == HcalBarrel) {
                minval = 0;
                maxval = nValidRHBarrel;
            }
            else if (subdet == HcalEndcap) {
                minval = nValidRHEndcap;
                maxval = nValidRHTotal;
            }
            else {
                printf("Rechit %u detId %u has invalid subdetector %u!\n", blockIdx.x, detId, subdet);
                return;
            }
        }
        __syncthreads();

        for (uint32_t j = threadIdx.x + minval; j < maxval && notDone; j += blockDim.x) {
            if (detId == rh_detIdRef[j]) {
                // Found it
                rh_detIdMap[i] = j;
                notDone = 0;
                //atomicAdd(&notDone, -1);
            //    break;
            }
            __syncthreads();
        }
    }
  }

  // Build detId map with 1 block per input rechit
  // Searches by detId for the matching index in lookup table, then terminates
  __global__ void buildDetIdMapPerBlock(uint32_t size,
                                uint32_t const* rh_detIdRef, // Lookup table index -> detId
                                uint32_t* rh_detIdMap,  //  Map for input rh detId -> lookup table index
                                uint32_t const* recHits_did) {  //  Rechit detIds
    
    __shared__ uint32_t detId, subdet, minval, maxval;

    if (threadIdx.x == 0) {
        detId = recHits_did[blockIdx.x];
        
        // Get subdetector encoded in detId
        // cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0048
        subdet = (detId >> 25) & 0x7;
        if (subdet == HcalBarrel) {
            minval = 0;
            maxval = nValidRHBarrel;
        }
        else if (subdet == HcalEndcap) {
            minval = nValidRHEndcap;
            maxval = nValidRHTotal;
        }
        else {
            printf("Rechit %u detId %u has invalid subdetector %u!\n", blockIdx.x, detId, subdet);
            return;
        }
    }
    __syncthreads();

    for (uint32_t j = threadIdx.x + minval; j < maxval; j += blockDim.x) {
        if (detId == rh_detIdRef[j]) {
            // Found it
            rh_detIdMap[blockIdx.x] = j;
            return;
        }
    }
  }
 
    __global__ void testDetIdMap(uint32_t size, 
                            uint32_t const* rh_detIdRef, // Lookup table index -> detId
                            uint32_t* rh_detIdMap,  //  Map for input rh detId -> lookup table index
                            uint32_t const* recHits_did) {  //  Rechit detIds 

        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        uint32_t detId = recHits_did[i];
        uint32_t index = rh_detIdMap[i];
        if (index >= nValidRHTotal || detId != rh_detIdRef[index])
            printf("Rechit %u detId %u MISMATCH with reference table index %u detId %u\n", i, detId, index, rh_detIdRef[index]);
    }


    
    void entryPoint(
        ::hcal::RecHitCollection<::calo::common::DevStoragePolicy> const& HBHERecHits_asInput,
        OutputPFRecHitDataGPU& HBHEPFRecHits_asOutput,
        PersistentDataGPU& persistentDataGPU,
        cudaStream_t cudaStream) {
      
      uint32_t nRH = HBHERecHits_asInput.size;   // Number of input rechits

#ifdef DEBUG_ENABLE
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, cudaStream);
#endif

      // First build the mapping for input rechits to lookup table indices
      buildDetIdMapPerBlock<<<nRH, 256, 0, cudaStream>>>(nRH, persistentDataGPU.rh_detId.get(), persistentDataGPU.rh_detIdMap.get(), HBHERecHits_asInput.did.get()); 

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);

      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("\nbuildDetIdMapPerBlock took %f ms\n\n", milliseconds);
      
      testDetIdMap<<<(nRH+127)/128, 128, 0, cudaStream>>>(nRH, persistentDataGPU.rh_detId.get(), persistentDataGPU.rh_detIdMap.get(), HBHERecHits_asInput.did.get()); 
#endif 

      // Fill PF rechit arrays
      convert_rechits_to_PFRechits<<<(nRH+31)/32, 128, 0, cudaStream>>>(
											 nRH,
                                             persistentDataGPU.rh_pos.get(),
                                             persistentDataGPU.rh_neighbours.get(),
                                             persistentDataGPU.rh_detIdMap.get(),
                                             HBHERecHits_asInput.energy.get(),
											 HBHERecHits_asInput.chi2.get(),
											 HBHERecHits_asInput.energyM0.get(),
											 HBHERecHits_asInput.timeM0.get(),
											 HBHERecHits_asInput.did.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_depth.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_layer.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_caloId.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_detId.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_time.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_energy.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_pt2.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_x.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_y.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_z.get());

    }
  } // namespace rechit 
} //  namespace pf

