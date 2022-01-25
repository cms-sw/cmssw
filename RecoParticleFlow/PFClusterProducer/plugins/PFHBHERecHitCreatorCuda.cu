// -*- c++ -*-
#include <Eigen/Dense>

#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/SimplePFGPUAlgos.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

// Uncomment for debug mode
#define DEBUG_ENABLE


namespace PFRecHit {
  namespace HCAL {
    
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

    __global__ void convert_rechits_to_PFRechits(const uint32_t nRHIn,
                         const uint32_t* nPFRHOut,
                         const bool* rh_mask,
                         const int* pfrhToInputIdx,
                         const int* inputToPFRHIdx,
                         const float3* rh_pos,
                         const int* rh_neighbours,
                         const int* rh_inputToFullIdx,
                         const int* rh_fullToInputIdx,
                         const float* recHits_energy,
						 const float* recHits_chi2,
						 const float* recHits_energyM0,
						 const float* recHits_timeM0,
						 const uint32_t* recHits_did,
						 int* pfrechits_depth,
						 int* pfrechits_layer,
						 int* pfrechits_detId,
						 float* pfrechits_time,
						 float* pfrechits_energy,
						 float* pfrechits_x,
						 float* pfrechits_y,
						 float* pfrechits_z,
                         int*   pfrechits_neighbours,
                         short* pfrechits_neighbourInfos) {
      

      for (uint32_t pfIdx = blockIdx.x * blockDim.x + threadIdx.x; pfIdx < nRHIn; pfIdx += blockDim.x*gridDim.x) {
          int i = pfrhToInputIdx[pfIdx]; // Get index corresponding to rechit input array
          if (i < 0) printf("convert kernel with pfIdx = %u has input index i = %u\n", pfIdx, i);
          pfrechits_time[pfIdx] = recHits_timeM0[i];
          float energy = recHits_energy[i];
          pfrechits_energy[pfIdx] = energy;
          
          uint32_t detid = recHits_did[i];
          pfrechits_detId[pfIdx] = detid;

          // cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0168
          pfrechits_depth[pfIdx] = (detid >> 20) & 0xf;

          // cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0050
          int subdet = (detid >> 25) & 0x7;
          int layer = 0;
          if (subdet == HcalBarrel)
            layer = PFLayer::HCAL_BARREL1;
          else if (subdet == HcalEndcap)
            layer = PFLayer::HCAL_ENDCAP;
          else
            printf("Invalid subdetector (%d) for detId %d\n", subdet, detid);
          
          pfrechits_layer[pfIdx] = layer;


          int index = rh_inputToFullIdx[i];  // Determine table index corresponding to this detId
          if (index < 0) printf("convert kernel with pfIdx = %u has full index = %u\n", pfIdx, index);
          float3 pos = rh_pos[index];   // position vector of this rechit
          pfrechits_x[pfIdx] = pos.x; 
          pfrechits_y[pfIdx] = pos.y; 
          pfrechits_z[pfIdx] = pos.z;
          
//          printf("Now on pfIdx %u\ti = %d\tindex = %d\tpos = (%f, %f, %f)\n", pfIdx, i, index, pos.x, pos.y, pos.z);
          
          auto associateNeighbour = [&] __device__ (uint32_t pos, uint32_t refPos, int eta, int phi, int depth) {
            int fullIdx = rh_neighbours[index*8+refPos];
            int inputIdx = fullIdx > -1 ? rh_fullToInputIdx[fullIdx] : -1;
            int pfrhIdx = inputIdx > -1 ? inputToPFRHIdx[inputIdx] : -1;
            short infos = pfrhIdx > -1 ? 0 : -1;
            if (pfrhIdx < 0 || pfrhIdx >= *nPFRHOut) {
                pfrechits_neighbours[pfIdx*8+pos] = -1;
                pfrechits_neighbourInfos[pfIdx*8+pos] = -1;
//                printf("Neigh %u has invalid pfrhIdx %d!\n", pos, pfrhIdx);
            }
            else {
                // Valid neighbour found. Compute neighbour infos 
                if (eta > 0) infos |= 1;
                infos |= (abs(eta) << 1);

                if (phi > 0) infos |= (1 << 4);
                infos |= (abs(phi) << 5);

                if (depth > 0) infos |= (1 << 8);
                infos |= (abs(depth) << 9);

                pfrechits_neighbours[pfIdx*8+pos] = pfrhIdx;
                pfrechits_neighbourInfos[pfIdx*8+pos] = infos;
//                printf("Neigh %u has pfrhIdx %d and infos %d\n", pos, pfrhIdx, infos);
            }
          };
          
          // Now fill neighbours and neighbourInfos
          // Reference neighbor array order:
          // SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
          associateNeighbour(0, 7,  0,  1, 0);   // N
          associateNeighbour(1, 0,  0, -1, 0);   // $
          associateNeighbour(2, 3,  1,  0, 0);   // E
          associateNeighbour(3, 4, -1,  0, 0);   // W
          associateNeighbour(4, 5,  1,  1, 0);   // NE
          associateNeighbour(5, 2, -1, -1, 0);   // SW
          associateNeighbour(6, 1,  1, -1, 0);   // SE
          associateNeighbour(7, 6, -1,  1, 0);   // NW

      }
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
                                int* rh_inputToFullIdx,  //  Map for input rh index -> lookup table index
                                int* rh_fullToInputIdx,  //  Map for lookup table index -> input rh index
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
            rh_inputToFullIdx[blockIdx.x] = j;
            rh_fullToInputIdx[j] = blockIdx.x;
            return;
        }
    }
  }
 
    __global__ void testDetIdMap(uint32_t size, 
                            const uint32_t* rh_detIdRef, // Lookup table index -> detId
                            const int* rh_inputToFullIdx,  //  Map for input rh detId -> lookup table index
                            const int* rh_fullToInputIdx,  //  Map for input rh detId -> lookup table index
                            const uint32_t* recHits_did) {  //  Rechit detIds 

        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        uint32_t detId = recHits_did[i];
        int index = rh_inputToFullIdx[i];
        int fullToInputIdx = index > -1 ? rh_fullToInputIdx[index] : -1;
        if (fullToInputIdx != i) {
            printf("Rechit %d detId %u doesn't match index from rh_fullToInputIdx %d!\n", i, detId, fullToInputIdx);
        }
        if (index >= nValidRHTotal || detId != rh_detIdRef[index])
            printf("Rechit %u detId %u MISMATCH with reference table index %u detId %u\n", i, detId, index, rh_detIdRef[index]);
    }
    
    __global__ void applyQTests(const uint32_t nRHIn,
                                bool* rh_mask,      // Mask for rechits by input index
                                const uint32_t* recHits_did,
                                const float* recHits_energy) {

        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nRHIn; i += gridDim.x * blockDim.x) {
            uint32_t detid = recHits_did[i];
            uint32_t subdet = (detid >> 25) & 0x7;
            uint32_t depth = (detid >> 20) & 0xf; 
            float threshold = 9999.;
            if (subdet == HcalBarrel) {
                if (depth == 1)
                    threshold = 0.1;
                else if (depth == 2)
                    threshold = 0.2;
                else if (depth == 3 || depth == 4)
                    threshold = 0.3;
                else
                    printf("i = %u\tInvalid depth %u for barrel rechit %u!\n", i, depth, detid);
                    
            }
            else if (subdet == HcalEndcap) {
                if (depth == 1)
                    threshold = 0.1;
                else if (depth >= 2 && depth <= 7)
                    threshold = 0.2;
                else
                    printf("i = %u\tInvalid depth %u for endcap rechit %u!\n", i, depth, detid);
            }
            else {
                printf("Rechit %u detId %u has invalid subdetector %u!\n", blockIdx.x, detid, subdet);
                return;
            }
            rh_mask[i] = (recHits_energy[i] >= threshold); 
        }
    }

    __global__ void applyMaskSerial(uint32_t nRHIn,
                              uint32_t* nPFRHOut,
                              const bool* rh_mask,
                              int* pfrhToInputIdx,
                              int* inputToPFRHIdx) {

        extern __shared__ uint16_t cleanedList[];
        __shared__ uint16_t cleanedTotal, pos;
        cleanedTotal = 0;

        pos = cleanedTotal = 0;
        for (uint16_t i = 0; i < nRHIn; i++) {
            if (rh_mask[i]) {
                pfrhToInputIdx[pos] = i;
                inputToPFRHIdx[i] = pos;
                pos++;
            }
            else {
                cleanedList[cleanedTotal] = i;
                cleanedTotal++;
            }
        }
        for (uint16_t i = 0; i < cleanedTotal; i++) {
            pfrhToInputIdx[pos+i] = cleanedList[i];
            inputToPFRHIdx[cleanedList[i]] = pos+i;
        }
        *nPFRHOut = pos;    // Total number of PFRecHits passing cuts
    }

    __global__ void initializeArrays(bool* rh_mask,
                                     int* rh_inputToFullIdx,
                                     int* rh_fullToInputIdx,
                                     int* pfrhToInputIdx,
                                     int* inputToPFRHIdx) {
        
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nValidRHTotal; i += blockDim.x * gridDim.x) {
            rh_mask[i] = true;
            //if (i == 35 || i == 39 || i == 69) rh_mask[i] = false;
            rh_inputToFullIdx[i] = -1;
            rh_fullToInputIdx[i] = -1;
            pfrhToInputIdx[i] = -1;
            inputToPFRHIdx[i] = -1;
        }
        // Test the rechit mask
//        if (blockIdx.x == 0 && threadIdx.x == 0) {
//            rh_mask[2] = false;
//            rh_mask[3] = false;
//        }
//        __syncthreads();
    }

    void entryPoint(
        ::hcal::RecHitCollection<::calo::common::DevStoragePolicy> const& HBHERecHits_asInput,
        OutputPFRecHitDataGPU& HBHEPFRecHits_asOutput,
        PersistentDataGPU& persistentDataGPU,
        ScratchDataGPU& scratchDataGPU,
        cudaStream_t cudaStream) {
      
      uint32_t nRHIn = HBHERecHits_asInput.size;   // Number of input rechits
      //uint32_t nRHIn = 1;   // Number of input rechits
      if (nRHIn == 0) {
        HBHEPFRecHits_asOutput.PFRecHits.size = 0;
        HBHEPFRecHits_asOutput.PFRecHits.sizeCleaned = 0;
        return;
      }

      uint32_t *h_nPFRHOut, *d_nPFRHOut;   // Number of output PFRecHits (total passing cuts)
      h_nPFRHOut = new uint32_t(0);
      cudaCheck(cudaMallocAsync(&d_nPFRHOut, sizeof(int), cudaStream));

#ifdef DEBUG_ENABLE
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaDeviceSynchronize();
      cudaEventRecord(start, cudaStream);
#endif

      initializeArrays<<<(scratchDataGPU.maxSize + 511)/512, 256, 0, cudaStream>>>(scratchDataGPU.rh_mask.get(), scratchDataGPU.rh_inputToFullIdx.get(), scratchDataGPU.rh_fullToInputIdx.get(), scratchDataGPU.pfrhToInputIdx.get(), scratchDataGPU.inputToPFRHIdx.get());
      cudaCheck(cudaGetLastError());

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);

      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("\ninitializeArrays took %f ms\n", milliseconds);
      cudaEventRecord(start, cudaStream);
#endif

      // First build the mapping for input rechits to lookup table indices
      buildDetIdMapPerBlock<<<nRHIn, 256, 0, cudaStream>>>(nRHIn, persistentDataGPU.rh_detId.get(), scratchDataGPU.rh_inputToFullIdx.get(), scratchDataGPU.rh_fullToInputIdx.get(), HBHERecHits_asInput.did.get()); 
      cudaCheck(cudaGetLastError());

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);

      milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("\nbuildDetIdMapPerBlock took %f ms\n", milliseconds);
      
      testDetIdMap<<<(nRHIn+127)/128, 128, 0, cudaStream>>>(nRHIn, persistentDataGPU.rh_detId.get(), scratchDataGPU.rh_inputToFullIdx.get(), scratchDataGPU.rh_fullToInputIdx.get(), HBHERecHits_asInput.did.get());
      cudaDeviceSynchronize();
      cudaCheck(cudaGetLastError());
      cudaEventRecord(start, cudaStream);
#endif
      
      applyQTests<<<(nRHIn+127)/128, 256, 0, cudaStream>>>(nRHIn, scratchDataGPU.rh_mask.get(), HBHERecHits_asInput.did.get(), HBHERecHits_asInput.energy.get());
      cudaCheck(cudaGetLastError());

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);

      milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("\napplyQTests took %f ms\n", milliseconds);
      cudaEventRecord(start, cudaStream);
#endif
      
      applyMaskSerial<<<1, 1, nRHIn * sizeof(short), cudaStream>>>(nRHIn, d_nPFRHOut, scratchDataGPU.rh_mask.get(), scratchDataGPU.pfrhToInputIdx.get(), scratchDataGPU.inputToPFRHIdx.get());
      cudaCheck(cudaGetLastError());

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("\napplyMask took %f ms\n\n", milliseconds);
#endif 
      cudaCheck(cudaMemcpyAsync(h_nPFRHOut, d_nPFRHOut, sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream));
      //cudaDeviceSynchronize();
//      cudaEvent_t sizeCopyEvt = 0;
//      cudaEventRecord(sizeCopyEvt, cudaStream);

      // Fill PF rechit arrays
      convert_rechits_to_PFRechits<<<(nRHIn+31)/32, 128, 0, cudaStream>>>(
            nRHIn,
            d_nPFRHOut,
            scratchDataGPU.rh_mask.get(),
            scratchDataGPU.pfrhToInputIdx.get(),
            scratchDataGPU.inputToPFRHIdx.get(),
            persistentDataGPU.rh_pos.get(),
            persistentDataGPU.rh_neighbours.get(),
            scratchDataGPU.rh_inputToFullIdx.get(),
            scratchDataGPU.rh_fullToInputIdx.get(),
            HBHERecHits_asInput.energy.get(),
            HBHERecHits_asInput.chi2.get(),
            HBHERecHits_asInput.energyM0.get(),
            HBHERecHits_asInput.timeM0.get(),
            HBHERecHits_asInput.did.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_depth.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_layer.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_detId.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_time.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_energy.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_x.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_y.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_z.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_neighbours.get(),
            HBHEPFRecHits_asOutput.PFRecHits.pfrh_neighbourInfos.get());
    
      cudaCheck(cudaGetLastError());

      // Make sure output size has finished copying before freeing memory
      //if (cudaEventQuery(sizeCopyEvt) != cudaSuccess) cudaEventSynchronize(sizeCopyEvt);
      if (cudaStreamQuery(cudaStream) != cudaSuccess) cudaCheck(cudaStreamSynchronize(cudaStream));
      HBHEPFRecHits_asOutput.PFRecHits.size = *h_nPFRHOut;
      HBHEPFRecHits_asOutput.PFRecHits.sizeCleaned = (nRHIn - *h_nPFRHOut);

      //HBHEPFRecHits_asOutput.PFRecHits.size = *h_nPFRHOut;
      cudaCheck(cudaFree(d_nPFRHOut));
      delete h_nPFRHOut;
    }
  } // namespace rechit 
} //  namespace pf

