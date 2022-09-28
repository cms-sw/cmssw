// -*- c++ -*-

#include <Eigen/Dense>

#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DeclsForKernels.h"
#include "SimplePFGPUAlgos.h"

// Uncomment for debug mode
//#define DEBUG_ENABLE

namespace PFRecHit {
  namespace HCAL {

    __constant__ uint32_t nValidRHBarrel;
    __constant__ uint32_t nValidRHEndcap;
    __constant__ uint32_t nValidRHTotal;
    __constant__ float qTestThresh;
    __constant__ uint32_t qTestDepthHB[4];
    __constant__ uint32_t qTestDepthHE[7];
    __constant__ float qTestThreshVsDepthHB[4];
    __constant__ float qTestThreshVsDepthHE[7];

    //
    // member methods:
    //  initializeCudaConstants [called from producer]
    //  initializeArrays
    //  buildDetIdMapPerBlockMulti (not used)
    //  buildDetIdMapPerBlock
    //  testDetIdMap (can be used for tesing maps)
    //  applyQTests (apply a single threshold)
    //  applyDepthThresholdQTests
    //  applyMaskSerial (simplier version)
    //  applyMask
    //  convert_rechits_to_PFRechits
    //  entryPoint [called from producer] utilizes:
    //   initializeArrays
    //   buildDetIdMapPerBlock
    //   applyDepthThresholdQTests
    //   applyMask
    //   convert_rechits_to_PFRechits

    void initializeCudaConstants(const PFRecHit::HCAL::Constants& cudaConstants) {

      cudaCheck(cudaMemcpyToSymbolAsync(nValidRHBarrel, &cudaConstants.nValidBarrelIds, sizeof(uint32_t)));
#ifdef DEBUG_ENABLE
      printf("--- HCAL Cuda constant values ---\n");
      uint32_t ival = 0;
      cudaCheck(cudaMemcpyFromSymbol(&ival, nValidRHBarrel, sizeof(uint32_t)));
      printf("nValidRHBarrel read from symbol: %u\n", ival);
#endif

      cudaCheck(cudaMemcpyToSymbolAsync(nValidRHEndcap, &cudaConstants.nValidEndcapIds, sizeof(uint32_t)));
#ifdef DEBUG_ENABLE
      ival = 0;
      cudaCheck(cudaMemcpyFromSymbol(&ival, nValidRHEndcap, sizeof(uint32_t)));
      printf("nValidRHEndcap read from symbol: %u\n", ival);
#endif

      //uint32_t total = &cudaConstants.nValidBarrelIds + &cudaConstants.nValidEndcapIds;
      cudaCheck(cudaMemcpyToSymbolAsync(nValidRHTotal, &cudaConstants.nValidDetIds, sizeof(uint32_t)));
#ifdef DEBUG_ENABLE
      ival = 0;
      cudaCheck(cudaMemcpyFromSymbol(&ival, nValidRHTotal, sizeof(uint32_t)));
      printf("nValidRHTotal read from symbol: %u\n", ival);
#endif

      cudaCheck(cudaMemcpyToSymbolAsync(qTestThresh, &cudaConstants.qTestThresh, sizeof(float)));
#ifdef DEBUG_ENABLE
      float val = 0;
      cudaCheck(cudaMemcpyFromSymbol(&val, qTestThresh, sizeof(float)));
      printf("qTestThresh read from symbol: %f\n\n", val);
#endif

      cudaCheck(cudaMemcpyToSymbolAsync(qTestDepthHB, &cudaConstants.qTestDepthHB, 4*sizeof(uint32_t)));
      cudaCheck(cudaMemcpyToSymbolAsync(qTestDepthHE, &cudaConstants.qTestDepthHE, 7*sizeof(uint32_t)));
      cudaCheck(cudaMemcpyToSymbolAsync(qTestThreshVsDepthHB, &cudaConstants.qTestThreshVsDepthHB, 4*sizeof(float)));
      cudaCheck(cudaMemcpyToSymbolAsync(qTestThreshVsDepthHE, &cudaConstants.qTestThreshVsDepthHE, 7*sizeof(float)));

    }

    // Initialize arrays used to store temporary values for each event
    __global__ void initializeArrays(uint32_t nRHIn,          // Number of input rechits
                                     int* rh_mask,            // Mask for input rechit index
                                     int* rh_inputToFullIdx,  // Mapping of input rechit index -> reference table index
                                     int* rh_fullToInputIdx,  // Mapping of reference table index -> input rechit index
                                     int* pfrhToInputIdx,     // Mapping of output PFRecHit index -> input rechit index
                                     int* inputToPFRHIdx) {   // Mapping of input rechit index -> output PFRecHit index

      // Reset mappings of reference table index. Total length = number of all valid HCAL detIds
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nValidRHTotal; i += blockDim.x * gridDim.x) {
        rh_inputToFullIdx[i] = -1;
        rh_fullToInputIdx[i] = -1;
      }

      // Reset mappings of input,output indices and rechit mask
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nRHIn; i += blockDim.x * gridDim.x) {
        pfrhToInputIdx[i] = -1;
        inputToPFRHIdx[i] = -1;
        rh_mask[i] = -2;
      }
    }

    __global__ void buildDetIdMapPerBlockMulti(
        uint32_t size,
        uint32_t const* rh_detIdRef,    // Reference table index -> detId
        uint32_t* rh_detIdMap,          // Map for input rechit detId -> reference table index
        uint32_t const* recHits_did) {  // Input rechit detIds

      __shared__ uint32_t detId, subdet, minval, maxval, notDone;

      for (uint32_t i = blockIdx.x; i < size; i += gridDim.x) {
        if (threadIdx.x == 0) {
          notDone = 1;
          detId = recHits_did[i];

          // Get subdetector encoded in detId
          // cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0048
          subdet = (detId >> DetId::kSubdetOffset) & DetId::kSubdetMask;
          if (subdet == HcalBarrel) {
            minval = 0;
            maxval = nValidRHBarrel;
          } else if (subdet == HcalEndcap) {
            minval = nValidRHEndcap;
            maxval = nValidRHTotal;
          } else {
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



    __global__ void buildDetIdMapHackathon(
        uint32_t size,
        uint32_t const* rh_detIdRef,    // Reference table index -> detId
        int* rh_inputToFullIdx,     // Map for input rechit detId -> reference table index
        int* rh_fullToInputIdx,     // Map for reference table index -> input rechit index    
        uint32_t const* recHits_did)    // Input rechit detIds
        {  

          int first = blockIdx.x*blockDim.x + threadIdx.x;
          for (int i = first; i < size; i += gridDim.x * blockDim.x) {
            auto detId = rh_detIdRef[i];
            for(int j = 0; j< size; ++j)
            {
              if(recHits_did[j] == detId)
              {
                rh_inputToFullIdx[j] = i;
                rh_fullToInputIdx[i] = j;
                return;
              }
            }
          }
        }

    // Build detId map with 1 block per input rechit
    // Searches by detId for the matching index in reference table
    __global__ void buildDetIdMapPerBlock(
        uint32_t size,                  // Number of input rechits
        uint32_t const* rh_detIdRef,    // Reference table index -> detId
        int* rh_inputToFullIdx,         // Map for input rechit index -> reference table index
        int* rh_fullToInputIdx,         // Map for reference table index -> input rechit index
        uint32_t const* recHits_did) {  // Input rechit detIds

      __shared__ uint32_t detId, subdet, minval, maxval;

      if (threadIdx.x == 0) {
        detId = recHits_did[blockIdx.x];

        // Get subdetector encoded in detId to narrow the range of reference table values to search
        // cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0048
        subdet = (detId >> DetId::kSubdetOffset) & DetId::kSubdetMask;
        if (subdet == HcalBarrel) {
          minval = 0;
          maxval = nValidRHBarrel;
        } else if (subdet == HcalEndcap) {
          minval = nValidRHEndcap;
          maxval = nValidRHTotal;
        } else {
          printf("Rechit %u detId %u has invalid subdetector %u!\n", blockIdx.x, detId, subdet);
          return;
        }
      }
      __syncthreads();

      // Search all valid rechits for matching detId
      for (uint32_t j = threadIdx.x + minval; j < maxval; j += blockDim.x) {
        if (detId == rh_detIdRef[j]) {
          // Found it
          rh_inputToFullIdx[blockIdx.x] = j;  // Input rechit index -> reference table index
          rh_fullToInputIdx[j] = blockIdx.x;  // Reference table index -> input rechit index
          return;
        }
      }
    }

    // Debugging function used to check the mapping of input index <-> reference table index
    __global__ void testDetIdMap(uint32_t size,                  // Number of input rechits
                                 const uint32_t* rh_detIdRef,    // Reference table index -> detId
                                 const int* rh_inputToFullIdx,   //  Map for input rh index -> reference table index
                                 const int* rh_fullToInputIdx,   //  Map for reference table index -> input rh index
                                 const uint32_t* recHits_did) {  //  Rechit detIds

      uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= size)
        return;

      uint32_t detId = recHits_did[i];
      int index = rh_inputToFullIdx[i];
      int fullToInputIdx = index > -1 ? rh_fullToInputIdx[index] : -1;
      if (fullToInputIdx != i) {
        printf("Rechit %d detId %u doesn't match index from rh_fullToInputIdx %d!\n", i, detId, fullToInputIdx);
      }
      if (index >= nValidRHTotal || detId != rh_detIdRef[index])
        printf(
            "Rechit %u detId %u MISMATCH with reference table index %u detId %u\n", i, detId, index, rh_detIdRef[index]);
    }

    // Phase 0 threshold test corresponding to PFRecHitQTestThreshold
    __global__ void applyQTests(const uint32_t nRHIn,
                                int* rh_mask,  // Mask for rechits by input index
                                const uint32_t* recHits_did,
                                const float* recHits_energy) {
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nRHIn; i += gridDim.x * blockDim.x) {
        rh_mask[i] = (recHits_energy[i] > qTestThresh);
      }
    }

    // Phase I threshold test corresponding to PFRecHitQTestHCALThresholdVsDepth
    __global__ void applyDepthThresholdQTests(const uint32_t nRHIn,           // Number of input rechits
                                              int* rh_mask,                   // Mask for rechit index
                                              const uint32_t* recHits_did,    // Input rechit detIds
                                              const float* recHits_energy) {  // Input rechit energy

      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nRHIn; i += gridDim.x * blockDim.x) {
        uint32_t detid = recHits_did[i];
        uint32_t subdet = (detid >> DetId::kSubdetOffset) & DetId::kSubdetMask;
        uint32_t depth = (detid >> HcalDetId::kHcalDepthOffset2) & HcalDetId::kHcalDepthMask2;
        float threshold = 9999.;
        if (subdet == HcalBarrel) {
	  bool found = false;
	  for (uint32_t j=0; j<4; j++){
	    if (depth == qTestDepthHB[j]){
	      threshold = qTestThreshVsDepthHB[j];
	      found = true; // found depth and threshold
	    }
	  }
	  if (!found)
            printf("i = %u\tInvalid depth %u for barrel rechit %u!\n", i, depth, detid);
        } else if (subdet == HcalEndcap) {
	  bool found = false;
	  for (uint32_t j=0; j<7; j++){
	    if (depth == qTestDepthHE[j]){
	      threshold = qTestThreshVsDepthHE[j];
	      found = true; // found depth and threshold
	    }
	  }
	  if (!found)
            printf("i = %u\tInvalid depth %u for endcap rechit %u!\n", i, depth, detid);
        } else {
          printf("Rechit %u detId %u has invalid subdetector %u!\n", blockIdx.x, detid, subdet);
          return;
        }
        // If this PFRecHit:
        //  Passes threshold cuts, set mask to 1
        //  Fails cuts and discarded, set mask to 0
        //  Should be cleaned (only applicable to HF), mask = -1 (default value)
        rh_mask[i] = (recHits_energy[i] >= threshold);
        if (rh_mask[i] < 0)
          printf("WARNING: Found input rechit %d has rh_mask = %d\n", i, rh_mask[i]);
      }
    }

    __global__ void applyMaskSerial(uint32_t nRHIn,
                                    uint32_t* nPFRHOut,
                                    //const bool* rh_mask,
                                    const int* rh_mask,
                                    int* pfrhToInputIdx,
                                    int* inputToPFRHIdx) {
      extern __shared__ uint16_t serial_cleanedList[];
      __shared__ uint16_t cleanedTotal, pos;

      pos = cleanedTotal = 0;
      for (uint16_t i = 0; i < nRHIn; i++) {
        if (rh_mask[i] == 1) {
          pfrhToInputIdx[pos] = i;
          inputToPFRHIdx[i] = pos;
          pos++;
        } else if (rh_mask[i] == -1) {
          serial_cleanedList[cleanedTotal] = i;
          cleanedTotal++;
        }
      }
      for (uint16_t i = 0; i < cleanedTotal; i++) {
        pfrhToInputIdx[pos + i] = serial_cleanedList[i];
        inputToPFRHIdx[serial_cleanedList[i]] = pos + i;
      }
      *nPFRHOut = pos;  // Total number of PFRecHits passing cuts
    }

    // Apply rechit mask and determine output PFRecHit ordering
    __global__ void applyMask(uint32_t nRHIn,          // Number of input rechits
                              uint32_t* nPFRHOut,      // Number of passing output PFRecHits
                              uint32_t* nPFRHCleaned,  // Number of cleaned output PFRecHits
                              const int* rh_mask,      // Rechit mask
                              int* pfrhToInputIdx,     // Mapping of output PFRecHit index -> input rechit index
                              int* inputToPFRHIdx) {   // Mapping of input rechit index -> output PFRecHit index

      extern __shared__ uint32_t cleanedList[];
      __shared__ uint32_t cleanedTotal, pos;

      if (threadIdx.x == 0) {
        pos = cleanedTotal = 0;
      }
      __syncthreads();

      for (uint32_t i = threadIdx.x; i < nRHIn; i += blockDim.x) {
        if (rh_mask[i] == 1) {  // Passing
          int k = atomicAdd(&pos, 1);
          pfrhToInputIdx[k] = i;
          inputToPFRHIdx[i] = k;
        } else if (rh_mask[i] == -1) {  // Cleaned
          int k = atomicAdd(&cleanedTotal, 1);
          cleanedList[k] = i;
        }
      }
      __syncthreads();

      // Loop over cleaned PFRecHits and append to the end of the output array
      for (uint32_t i = threadIdx.x; i < cleanedTotal; i += blockDim.x) {
        pfrhToInputIdx[pos + i] = cleanedList[i];
        inputToPFRHIdx[cleanedList[i]] = pos + i;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        *nPFRHOut = pos;               // Total number of PFRecHits passing cuts
        *nPFRHCleaned = cleanedTotal;  // Total number of cleaned PFRecHits
      }
    }

    // Fill output PFRecHit arrays
    __global__ void convert_rechits_to_PFRechits(const uint32_t nRHIn,
                                                 const uint32_t* nPFRHOut,
                                                 const uint32_t* nPFRHCleaned,
                                                 const int* rh_mask,
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
                                                 int* pfrechits_neighbours,
                                                 short* pfrechits_neighbourInfos) {

      for (uint32_t pfIdx = blockIdx.x * blockDim.x + threadIdx.x; pfIdx < (*nPFRHOut + *nPFRHCleaned);
           pfIdx += blockDim.x * gridDim.x) {

        int i = pfrhToInputIdx[pfIdx];  // Get input rechit index corresponding to output PFRecHit index pfIdx
        if (i < 0)
          printf("convert kernel with pfIdx = %u has input index i = %u\n", pfIdx, i);
        pfrechits_time[pfIdx] = recHits_timeM0[i];
        float energy = recHits_energy[i];
        pfrechits_energy[pfIdx] = energy;

        uint32_t detid = recHits_did[i];
        pfrechits_detId[pfIdx] = detid;

        //bool debug = (detid == 1158706177) ? true : false;
        bool debug = false;
        // cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0168
        pfrechits_depth[pfIdx] = (detid >> HcalDetId::kHcalDepthOffset2) & HcalDetId::kHcalDepthMask2;

        // cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0050
        int subdet = (detid >> DetId::kSubdetOffset) & DetId::kSubdetMask;
        int layer = 0;
        if (subdet == HcalBarrel)
          layer = PFLayer::HCAL_BARREL1;
        else if (subdet == HcalEndcap)
          layer = PFLayer::HCAL_ENDCAP;
        else
          printf("Invalid subdetector (%d) for detId %d: pfIdx = %d\tinputIdx = %d\tfullIdx = %d\n",
                 subdet,
                 detid,
                 pfIdx,
                 i,
                 rh_inputToFullIdx[i]);

        pfrechits_layer[pfIdx] = layer;

        int index = rh_inputToFullIdx[i];  // Determine reference table index corresponding to this input index
        if (index < 0)
          printf("convert kernel with pfIdx = %u has full index = %u\n", pfIdx, index);
        float3 pos = rh_pos[index];  // position vector of this rechit
        pfrechits_x[pfIdx] = pos.x;
        pfrechits_y[pfIdx] = pos.y;
        pfrechits_z[pfIdx] = pos.z;

        if (debug) {
          printf("Now debugging rechit %d\tpfIdx %u\ti = %d\tindex = %d\tpos = (%f, %f, %f)\n",
                 detid,
                 pfIdx,
                 i,
                 index,
                 pos.x,
                 pos.y,
                 pos.z);
          printf("\trh_neighbours = [%d, %d, %d, %d, %d, %d, %d, %d]\n\n",
                 rh_neighbours[index * 8],
                 rh_neighbours[index * 8 + 1],
                 rh_neighbours[index * 8 + 2],
                 rh_neighbours[index * 8 + 3],
                 rh_neighbours[index * 8 + 4],
                 rh_neighbours[index * 8 + 5],
                 rh_neighbours[index * 8 + 6],
                 rh_neighbours[index * 8 + 7]);
        }

        // Lambda function for filling PFRecHit neighbour arrays
        // pos: Order in PFRecHit neighbour array. First four values correspond to 4-neighbours: N,S,E,W
        // refPos: Order of rechit neighbors given in neighboursHcal_ array from PFHCALDenseIdNavigator
        // eta: ieta for this direction relative to center
        // phi: iphi for this direction relative to center
        // depth: idepth for this direction relative to center (always 0 for layer clusters)
        auto associateNeighbour = [&] __device__(uint32_t pos, uint32_t refPos, int eta, int phi, int depth) {
          int fullIdx = rh_neighbours[index * 8 + refPos];                // Reference table index for this neighbour
          int inputIdx = fullIdx > -1 ? rh_fullToInputIdx[fullIdx] : -1;  // Input rechit index for this neighbour
          int pfrhIdx = inputIdx > -1 ? inputToPFRHIdx[inputIdx] : -1;    // Output PFRecHit index for this neighbour
          short infos = pfrhIdx > -1 ? 0 : -1;
          if (debug)
            printf(
                "associateNeighbour for rechit %d pos %d refPos %d: fullIdx = %d%sinputIdx = %d\tpfrhIdx = "
                "%d\trecHits_did[inputIdx] = %d\n",
                detid,
                pos,
                refPos,
                fullIdx,
                (fullIdx == 0) ? "\t\t" : "\t",
                inputIdx,
                pfrhIdx,
                recHits_did[inputIdx]);
          if (pfrhIdx < 0 ||
              pfrhIdx >= *nPFRHOut) {  // Only include valid PFRecHit indices. Don't include cleaned rechits
            pfrechits_neighbours[pfIdx * 8 + pos] = -1;
            pfrechits_neighbourInfos[pfIdx * 8 + pos] = -1;
            if (debug)
              printf("\tNeigh %u has invalid pfrhIdx %d!\n", pos, pfrhIdx);
          } else {
            // Valid neighbour found. Compute neighbour infos
            if (eta > 0)
              infos |= 1;
            infos |= (abs(eta) << 1);

            if (phi > 0)
              infos |= (1 << 4);
            infos |= (abs(phi) << 5);

            if (depth > 0)
              infos |= (1 << 8);
            infos |= (abs(depth) << 9);

            // Set PFRecHit index and infos for this neighbour
            pfrechits_neighbours[pfIdx * 8 + pos] = pfrhIdx;
            pfrechits_neighbourInfos[pfIdx * 8 + pos] = infos;
            if (debug)
              printf("\tNeigh %u has pfrhIdx %d and infos %d\n", pos, pfrhIdx, infos);
          }
        };

        // Now fill neighbours and neighbourInfos
        // Reference neighbor array order from navigator:
        // SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
        associateNeighbour(0, 7, 0, 1, 0);    // N
        associateNeighbour(1, 0, 0, -1, 0);   // $
        associateNeighbour(2, 3, 1, 0, 0);    // E
        associateNeighbour(3, 4, -1, 0, 0);   // W
        associateNeighbour(4, 5, 1, 1, 0);    // NE
        associateNeighbour(5, 2, -1, -1, 0);  // SW
        associateNeighbour(6, 1, 1, -1, 0);   // SE
        associateNeighbour(7, 6, -1, 1, 0);   // NW
      }
    }

    void entryPoint(::hcal::RecHitCollection<::calo::common::DevStoragePolicy> const& HBHERecHits_asInput,
                    OutputPFRecHitDataGPU& HBHEPFRecHits_asOutput,
                    PersistentDataGPU& persistentDataGPU,
                    ScratchDataGPU& scratchDataGPU,
                    cudaStream_t cudaStream,
                    std::array<float, 5>& timer) {

      uint32_t nRHIn = HBHERecHits_asInput.size;  // Number of input rechits
      if (nRHIn == 0) {
        HBHEPFRecHits_asOutput.PFRecHits.size = 0;
        HBHEPFRecHits_asOutput.PFRecHits.sizeCleaned = 0;
        return;
      }

      // uint32_t *h_nPFRHOut, *d_nPFRHOut;          // Number of output PFRecHits (total passing cuts)
      // uint32_t *h_nPFRHCleaned, *d_nPFRHCleaned;  // Number of cleaned PFRecHits
      // h_nPFRHOut = new uint32_t(0);
      // h_nPFRHCleaned = new uint32_t(0);
      // cudaCheck(cudaMallocAsync(&d_nPFRHOut, sizeof(int), cudaStream));
      // cudaCheck(cudaMallocAsync(&d_nPFRHCleaned, sizeof(int), cudaStream));


      cms::cuda::device::unique_ptr<uint32_t[]> d_nPFRHOut; // Number of output PFRecHits (total passing cuts)
      cms::cuda::device::unique_ptr<uint32_t[]> d_nPFRHCleaned; // Number of cleaned PFRecHits
      cms::cuda::host::unique_ptr<uint32_t[]> h_nPFRHOut;
      cms::cuda::host::unique_ptr<uint32_t[]> h_nPFRHCleaned;
      
      d_nPFRHOut = cms::cuda::make_device_unique<uint32_t[]>(sizeof(uint32_t) , cudaStream);
      d_nPFRHCleaned = cms::cuda::make_device_unique<uint32_t[]>(sizeof(uint32_t) , cudaStream);

      h_nPFRHOut = cms::cuda::make_host_unique<uint32_t[]>(sizeof(uint32_t) , cudaStream);
      h_nPFRHCleaned = cms::cuda::make_host_unique<uint32_t[]>(sizeof(uint32_t) , cudaStream);

#ifdef DEBUG_ENABLE
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaDeviceSynchronize();
      cudaEventRecord(start, cudaStream);
#endif
      int threadsPerBlock = 256;
      // Initialize scratch arrays
      initializeArrays<<<(scratchDataGPU.maxSize + threadsPerBlock-1) / threadsPerBlock, threadsPerBlock, 0, cudaStream>>>(
          nRHIn,
          scratchDataGPU.rh_mask.get(),
          scratchDataGPU.rh_inputToFullIdx.get(),
          scratchDataGPU.rh_fullToInputIdx.get(),
          scratchDataGPU.pfrhToInputIdx.get(),
          scratchDataGPU.inputToPFRHIdx.get());
      cudaCheck(cudaGetLastError());

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&timer[0], start, stop);
      printf("\ninitializeArrays took %f ms\n", timer[0]);
      cudaEventRecord(start, cudaStream);
#endif

      // // First build the mapping for input rechits to reference table indices
      buildDetIdMapPerBlock<<<nRHIn, 256, 0, cudaStream>>>(nRHIn,
                                                           persistentDataGPU.rh_detId.get(),
                                                           scratchDataGPU.rh_inputToFullIdx.get(),
                                                           scratchDataGPU.rh_fullToInputIdx.get(),
                                                           HBHERecHits_asInput.did.get());
      cudaCheck(cudaGetLastError());

      // First build the mapping for input rechits to reference table indices
      // buildDetIdMapHackathon<<<(nRHIn + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, cudaStream>>>(nRHIn,
      //                                                      persistentDataGPU.rh_detId.get(),
      //                                                      scratchDataGPU.rh_inputToFullIdx.get(),
      //                                                      scratchDataGPU.rh_fullToInputIdx.get(),
      //                                                      HBHERecHits_asInput.did.get());
      // cudaCheck(cudaGetLastError());


    // Debugging function used to check the mapping of input index <-> reference table index
    // testDetIdMap<<<(nRHIn + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, cudaStream>>>(nRHIn,
    //                                                        persistentDataGPU.rh_detId.get(),
    //                                                        scratchDataGPU.rh_inputToFullIdx.get(),
    //                                                        scratchDataGPU.rh_fullToInputIdx.get(),
    //                                                        HBHERecHits_asInput.did.get());
     cudaCheck(cudaGetLastError());
#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&timer[1], start, stop);
      printf("\nbuildDetIdMapPerBlock took %f ms\n", timer[1]);

      cudaEventRecord(start, cudaStream);
#endif

      // Apply PFRecHit threshold & quality tests

      //applyQTests<<<(nRHIn+127)/128, 256, 0, cudaStream>>>(nRHIn, scratchDataGPU.rh_mask.get(), HBHERecHits_asInput.did.get(), HBHERecHits_asInput.energy.get());
      
      applyDepthThresholdQTests<<<(nRHIn + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, cudaStream>>>(
          nRHIn, scratchDataGPU.rh_mask.get(), HBHERecHits_asInput.did.get(), HBHERecHits_asInput.energy.get());
      cudaCheck(cudaGetLastError());

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&timer[2], start, stop);
      printf("\napplyQTests took %f ms\n", timer[2]);
      cudaEventRecord(start, cudaStream);
#endif

      // Apply rechit mask and determine output PFRecHit order
      applyMask<<<1, 256, nRHIn * sizeof(int), cudaStream>>>(nRHIn,
                                                             d_nPFRHOut.get(),
                                                             d_nPFRHCleaned.get(),
                                                             scratchDataGPU.rh_mask.get(),
                                                             scratchDataGPU.pfrhToInputIdx.get(),
                                                             scratchDataGPU.inputToPFRHIdx.get());
      cudaCheck(cudaGetLastError());

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&timer[3], start, stop);
      printf("\napplyMask took %f ms\n\n", timer[3]);
#endif
      cudaCheck(cudaMemcpyAsync(h_nPFRHOut.get(), d_nPFRHOut.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream));
      cudaCheck(cudaMemcpyAsync(h_nPFRHCleaned.get(), d_nPFRHCleaned.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream));

#ifdef DEBUG_ENABLE
      cudaDeviceSynchronize();
      cudaEventRecord(start);
#endif

      // Fill output PFRecHit arrays
      convert_rechits_to_PFRechits<<<(nRHIn + 31) / 32, 128, 0, cudaStream>>>(
          nRHIn,
          d_nPFRHOut.get(),
          d_nPFRHCleaned.get(),
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
      if (cudaStreamQuery(cudaStream) != cudaSuccess)
        cudaCheck(cudaStreamSynchronize(cudaStream));

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&timer[4], start, stop);
      printf("\nconvert_rechits_to_PFRechits took %f ms\n\n", timer[4]);
#endif

      HBHEPFRecHits_asOutput.PFRecHits.size = *(h_nPFRHOut.get());
      HBHEPFRecHits_asOutput.PFRecHits.sizeCleaned = *(h_nPFRHCleaned.get());

      // cudaCheck(cudaFree(d_nPFRHOut));
      // cudaCheck(cudaFree(d_nPFRHCleaned));
      // delete h_nPFRHOut;
      // delete h_nPFRHCleaned;
    }
  }  // namespace HCAL
}  //  namespace PFRecHit
