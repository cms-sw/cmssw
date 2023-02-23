// -*- c++ -*-

#include <Eigen/Dense>

#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "DeclsForKernels.h"
#include "SimplePFGPUAlgos.h"

// Uncomment for debug mode
//#define DEBUG_ENABLE

namespace PFRecHit {
  namespace HCAL {

    // member methods:
    //  initializeArrays
    //  buildDetIdMap
    //  applyDepthThresholdQTestsAndMask
    //  convert_rechits_to_PFRechits
    //  entryPoint [called from producer] utilizes:
    //   initializeArrays
    //   buildDetIdMapPerBlock
    //   applyDepthThresholdQTestsAndMask
    //   convert_rechits_to_PFRechits

    // some constants
    constexpr int maxDepthHB = 4;
    constexpr int maxDepthHE = 7;
    constexpr int firstHBRing = 1;
    constexpr int lastHBRing = 16;
    constexpr int firstHERing = 16;
    constexpr int lastHERing = 29;
    constexpr int IPHI_MAX = 72;

    // Initialize arrays used to store temporary values for each event
    __global__ void initializeArrays(uint32_t nTopoArraySize,  // Takes detId.size() but needs work
                                     uint32_t nRHIn,           // Number of input rechits
                                     int* rh_inputToFullIdx,   // Mapping of input rechit index -> reference table index
                                     int* rh_fullToInputIdx,   // Mapping of reference table index -> input rechit index
                                     int* pfrhToInputIdx,      // Mapping of output PFRecHit index -> input rechit index
                                     int* inputToPFRHIdx) {    // Mapping of input rechit index -> output PFRecHit index

      // Reset mappings of reference table index. Total length = number of all valid HCAL detIds
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nTopoArraySize; i += blockDim.x * gridDim.x) {
        rh_fullToInputIdx[i] = -1;
        rh_inputToFullIdx[i] = -1;
      }

      // Reset mappings of input,output indices and rechit mask
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nRHIn; i += blockDim.x * gridDim.x) {
        pfrhToInputIdx[i] = -1;
        inputToPFRHIdx[i] = -1;
      }
    }

    // Get subdetector encoded in detId to narrow the range of reference table values to search
    // cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0048
    __device__ uint32_t getSubdet(uint32_t detId) { return ((detId >> DetId::kSubdetOffset) & DetId::kSubdetMask); }

    //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0163
    __device__ uint32_t getDepth(uint32_t detId) {
      return ((detId >> HcalDetId::kHcalDepthOffset2) & HcalDetId::kHcalDepthMask2);
    }

    //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0148
    __device__ uint32_t getIetaAbs(uint32_t detId) {
      return ((detId >> HcalDetId::kHcalEtaOffset2) & HcalDetId::kHcalEtaMask2);
    }

    //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0157
    __device__ uint32_t getIphi(uint32_t detId) { return (detId & HcalDetId::kHcalPhiMask2); }

    //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0141
    __device__ int getZside(uint32_t detId) { return ((detId & HcalDetId::kHcalZsideMask2) ? (1) : (-1)); }

    //https://cmssdt.cern.ch/lxr/source/Geometry/CaloTopology/src/HcalTopology.cc#1170
    __device__ uint32_t detId2denseIdHB(uint32_t detId) {
      const int nEtaHB = (lastHBRing - firstHBRing + 1);
      const int ip = getIphi(detId);
      const int ie = getIetaAbs(detId);
      const int dp = getDepth(detId);
      const int zn = getZside(detId);
      unsigned int retval = 0xFFFFFFFFu;
      retval = (dp - 1) + maxDepthHB * (ip - 1);
      if (zn > 0)
        retval += maxDepthHB * IPHI_MAX * (ie * zn - firstHBRing);
      else
        retval += maxDepthHB * IPHI_MAX * (ie * zn + lastHBRing + nEtaHB);

      return retval;
    }

    //https://cmssdt.cern.ch/lxr/source/Geometry/CaloTopology/src/HcalTopology.cc#1189
    __device__ uint32_t detId2denseIdHE(uint32_t detId) {
      const int nEtaHE = (lastHERing - firstHERing + 1);
      const int maxPhiHE = IPHI_MAX;
      const int ip = getIphi(detId);
      const int ie = getIetaAbs(detId);
      const int dp = getDepth(detId);
      const int zn = getZside(detId);
      unsigned int retval = 0xFFFFFFFFu;
      const int HBSize = maxDepthHB * 16 * IPHI_MAX * 2;
      retval = (dp - 1) + maxDepthHE * (ip - 1) + HBSize;
      if (zn > 0)
        retval += maxDepthHE * maxPhiHE * (ie * zn - firstHERing);
      else
        retval += maxDepthHE * maxPhiHE * (ie * zn + lastHERing + nEtaHE);

      return retval;
    }

    __device__ uint32_t detId2denseId(uint32_t detId) {
      if (getSubdet(detId) == HcalBarrel)
        return detId2denseIdHB(detId);
      else if (getSubdet(detId) == HcalEndcap)
        return detId2denseIdHE(detId);
      else
        printf("invalid detId\n");
    }

    __global__ void buildDetIdMap(uint32_t size,
                                  uint32_t const* denseIdarr,     // denseId array
                                  uint32_t const* detId,          // Takes in topoDataProduct.detId
                                  int* rh_inputToFullIdx,         // Map for input rechit detId -> reference table index
                                  int* rh_fullToInputIdx,         // Map for reference table index -> input rechit index
                                  uint32_t const* recHits_did) {  // Input rechit detIds

      int first = blockIdx.x * blockDim.x + threadIdx.x;
      for (int i = first; i < size; i += gridDim.x * blockDim.x) {
        // i: index for input rechits
        auto detId = recHits_did[i];
        auto denseId = detId2denseId(detId);
        auto fullIdx = denseId - denseIdarr[0];
        rh_inputToFullIdx[i] = fullIdx;  // Input rechit index -> reference table index
        rh_fullToInputIdx[fullIdx] = i;  // Reference table index -> input rechit index
      }
    }

    // Phase I threshold test corresponding to PFRecHitQTestHCALThresholdVsDepth
    // Apply rechit mask and determine output PFRecHit ordering
    __global__ void applyDepthThresholdQTestsAndMask(
        const uint32_t nRHIn,  // Number of input rechits
        int const* nDepthHB,
        int const* nDepthHE,
        int const* depthHB,  // The following from recHitParamsProduct
        int const* depthHE,
        float const* thresholdE_HB,
        float const* thresholdE_HE,
        const uint32_t* recHits_did,  // Input rechit detIds
        const float* recHits_energy,  // Input rechit energy
        uint32_t* nPFRHOut,           // Number of passing output PFRecHits
        uint32_t* nPFRHCleaned,       // Number of cleaned output PFRecHits
        int* pfrhToInputIdx,          // Mapping of output PFRecHit index -> input rechit index
        int* inputToPFRHIdx) {        // Mapping of input rechit index -> output PFRecHit index

      extern __shared__ uint32_t cleanedList[];
      __shared__ uint32_t cleanedTotal, pos;

      if (threadIdx.x == 0) {
        pos = cleanedTotal = 0;
      }
      __syncthreads();
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nRHIn; i += gridDim.x * blockDim.x) {
        uint32_t detid = recHits_did[i];
        uint32_t subdet = (detid >> DetId::kSubdetOffset) & DetId::kSubdetMask;
        uint32_t depth = (detid >> HcalDetId::kHcalDepthOffset2) & HcalDetId::kHcalDepthMask2;
        float threshold = 9999.;
        if (subdet == HcalBarrel) {
          bool found = false;
          for (uint32_t j = 0; j < *nDepthHB; j++) {
            if (depth == depthHB[j]) {
              threshold = thresholdE_HB[j];
              found = true;  // found depth and threshold
            }
          }
          if (!found)
            printf("i = %u\tInvalid depth %u for barrel rechit %u!\n", i, depth, detid);
        } else if (subdet == HcalEndcap) {
          bool found = false;
          for (uint32_t j = 0; j < *nDepthHE; j++) {
            if (depth == depthHE[j]) {
              threshold = thresholdE_HE[j];
              found = true;  // found depth and threshold
            }
          }
          if (!found)
            printf("i = %u\tInvalid depth %u for endcap rechit %u!\n", i, depth, detid);
        } else {
          printf("Rechit %u detId %u has invalid subdetector %u!\n", blockIdx.x, detid, subdet);
          return;
        }

        if (recHits_energy[i] >= threshold) {  // Passing
          int k = atomicAdd(&pos, 1);
          pfrhToInputIdx[k] = i;
          inputToPFRHIdx[i] = k;
        } else if (false) {  // Cleaned
          int k = atomicAdd(&cleanedTotal, 1);
          cleanedList[k] = i;
        }
      }
      __syncthreads();

      // Loop over cleaned PFRecHits and append to the end of the output array
      //for (uint32_t i = threadIdx.x; i < cleanedTotal; i += blockDim.x) {
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cleanedTotal; i += gridDim.x * blockDim.x) {
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
                                                 const uint32_t* offset,
                                                 const uint32_t* nPFRHOut,
                                                 const int* pfrhToInputIdx,
                                                 const int* inputToPFRHIdx,
                                                 const float3* position,
                                                 const int* neighbours,
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
                                                 const bool associate_neighbours) {
      for (uint32_t pfIdx = blockIdx.x * blockDim.x + threadIdx.x + (offset == nullptr ? 0 : *offset);
           pfIdx < (*nPFRHOut + (offset == nullptr ? 0 : *offset));
           pfIdx += blockDim.x * gridDim.x) {
        int i = pfrhToInputIdx[pfIdx];  // Get input rechit index corresponding to output PFRecHit index pfIdx
        if (i < 0)
          printf("convert kernel with pfIdx = %u has input index i = %u\n", pfIdx, i);
        pfrechits_time[pfIdx] = recHits_timeM0[i];
        pfrechits_energy[pfIdx] = recHits_energy[i];

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
        float3 pos = position[index];  // position vector of this rechit
        pfrechits_x[pfIdx] = pos.x;
        pfrechits_y[pfIdx] = pos.y;
        pfrechits_z[pfIdx] = pos.z;

        if (!associate_neighbours)
          continue;

        if (debug)
          printf("Now debugging rechit %d\tpfIdx %u\ti = %d\tindex = %d\tpos = (%f, %f, %f)\n",
                 detid,
                 pfIdx,
                 i,
                 index,
                 pos.x,
                 pos.y,
                 pos.z);

        // Lambda function for filling PFRecHit neighbour arrays
        // pos: Order in PFRecHit neighbour array. First four values correspond to 4-neighbours: N,S,E,W
        // refPos: Order of rechit neighbors given in neighboursHcal_ array from PFHCALDenseIdNavigator
        // eta: ieta for this direction relative to center
        // phi: iphi for this direction relative to center
        // depth: idepth for this direction relative to center (always 0 for layer clusters)
        auto associateNeighbour = [&] __device__(uint32_t pos, uint32_t refPos, int eta, int phi, int depth) {
          int fullIdx = neighbours[index * 8 + refPos];                   // Reference table index for this neighbour
          int inputIdx = fullIdx > -1 ? rh_fullToInputIdx[fullIdx] : -1;  // Input rechit index for this neighbour
          int pfrhIdx = inputIdx > -1 ? inputToPFRHIdx[inputIdx] : -1;    // Output PFRecHit index for this neighbour
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
          if (pfrhIdx >= 0) {  // Only include valid PFRecHit indices.
            // Set PFRecHit index and infos for this neighbour
            pfrechits_neighbours[pfIdx * 8 + pos] = pfrhIdx;
            if (debug)
              printf("\tNeigh %u has pfrhIdx %d.\n", pos, pfrhIdx);
          } else {
            pfrechits_neighbours[pfIdx * 8 + pos] = -1;
            if (debug)
              printf("\tNeigh %u has invalid pfrhIdx %d!\n", pos, pfrhIdx);
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
                    const ConstantProducts& constantProducts,
                    OutputPFRecHitDataGPU& HBHEPFRecHits_asOutput,
                    ScratchDataGPU& scratchDataGPU,
                    cudaStream_t cudaStream,
                    std::array<float, 5>& timer) {
      bool debug = false;
      if (debug) {
        std::cout << constantProducts.denseId.size() << std::endl;
        std::cout << constantProducts.detId.size() << std::endl;
        std::cout << constantProducts.position.size() << std::endl;
        std::cout << constantProducts.neighbours.size() << std::endl;
      }
      uint32_t nRHIn = HBHERecHits_asInput.size;  // Number of input rechits
      if (nRHIn == 0) {
        HBHEPFRecHits_asOutput.PFRecHits.size = 0;
        HBHEPFRecHits_asOutput.PFRecHits.sizeCleaned = 0;
        return;
      }

      cms::cuda::device::unique_ptr<uint32_t[]> d_nPFRHOut;      // Number of output PFRecHits (total passing cuts)
      cms::cuda::device::unique_ptr<uint32_t[]> d_nPFRHCleaned;  // Number of cleaned PFRecHits
      cms::cuda::host::unique_ptr<uint32_t[]> h_nPFRHOut;
      cms::cuda::host::unique_ptr<uint32_t[]> h_nPFRHCleaned;

      d_nPFRHOut = cms::cuda::make_device_unique<uint32_t[]>(sizeof(uint32_t), cudaStream);
      d_nPFRHCleaned = cms::cuda::make_device_unique<uint32_t[]>(sizeof(uint32_t), cudaStream);

      h_nPFRHOut = cms::cuda::make_host_unique<uint32_t[]>(sizeof(uint32_t), cudaStream);
      h_nPFRHCleaned = cms::cuda::make_host_unique<uint32_t[]>(sizeof(uint32_t), cudaStream);

#ifdef DEBUG_ENABLE
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaDeviceSynchronize();
      cudaEventRecord(start, cudaStream);
#endif
      int threadsPerBlock = 256;
      // Initialize scratch arrays
      initializeArrays<<<(max(scratchDataGPU.maxSize, (int)constantProducts.detId.size()) + threadsPerBlock - 1) /
                             threadsPerBlock,
                         threadsPerBlock,
                         0,
                         cudaStream>>>(constantProducts.detId.size(),
                                       nRHIn,
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

      // First build the mapping for input rechits to reference table indices
      buildDetIdMap<<<(nRHIn + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, cudaStream>>>(
          nRHIn,
          constantProducts.topoDataProduct.denseId,
          constantProducts.topoDataProduct.detId,
          scratchDataGPU.rh_inputToFullIdx.get(),
          scratchDataGPU.rh_fullToInputIdx.get(),
          HBHERecHits_asInput.did.get());
      cudaCheck(cudaGetLastError());

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&timer[1], start, stop);
      printf("\nbuildDetIdMapPerBlock took %f ms\n", timer[1]);

      cudaEventRecord(start, cudaStream);
#endif

      // Apply PFRecHit threshold & quality tests
      // Apply rechit mask and determine output PFRecHit order
      applyDepthThresholdQTestsAndMask<<<1, threadsPerBlock, 0, cudaStream>>>(
          nRHIn,
          constantProducts.recHitParametersProduct.nDepthHB,
          constantProducts.recHitParametersProduct.nDepthHE,
          constantProducts.recHitParametersProduct.depthHB,
          constantProducts.recHitParametersProduct.depthHE,
          constantProducts.recHitParametersProduct.thresholdE_HB,
          constantProducts.recHitParametersProduct.thresholdE_HE,
          HBHERecHits_asInput.did.get(),
          HBHERecHits_asInput.energy.get(),
          d_nPFRHOut.get(),
          d_nPFRHCleaned.get(),
          scratchDataGPU.pfrhToInputIdx.get(),
          scratchDataGPU.inputToPFRHIdx.get());
      cudaCheck(cudaGetLastError());

#ifdef DEBUG_ENABLE
      cudaEventRecord(stop, cudaStream);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&timer[3], start, stop);
      printf("\napplyDepthThresholdQTestsAndMask took %f ms\n\n", timer[3]);
#endif

      cms::cuda::copyAsync(h_nPFRHOut, d_nPFRHOut, sizeof(uint32_t), cudaStream);
      cms::cuda::copyAsync(h_nPFRHCleaned, d_nPFRHCleaned, sizeof(uint32_t), cudaStream);

#ifdef DEBUG_ENABLE
      cudaDeviceSynchronize();
      cudaEventRecord(start);
#endif

      // Fill output PFRecHit arrays
      convert_rechits_to_PFRechits<<<(nRHIn + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, cudaStream>>>(
          nRHIn,
          nullptr,
          d_nPFRHOut.get(),
          scratchDataGPU.pfrhToInputIdx.get(),
          scratchDataGPU.inputToPFRHIdx.get(),
          constantProducts.topoDataProduct.position,
          constantProducts.topoDataProduct.neighbours,
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
          true);
      cudaCheck(cudaGetLastError());
      convert_rechits_to_PFRechits<<<(nRHIn + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, cudaStream>>>(
          nRHIn,
          d_nPFRHOut.get(),
          d_nPFRHCleaned.get(),
          scratchDataGPU.pfrhToInputIdx.get(),
          scratchDataGPU.inputToPFRHIdx.get(),
          constantProducts.topoDataProduct.position,
          constantProducts.topoDataProduct.neighbours,
          scratchDataGPU.rh_inputToFullIdx.get(),
          scratchDataGPU.rh_fullToInputIdx.get(),
          HBHERecHits_asInput.energy.get(),
          HBHERecHits_asInput.chi2.get(),
          HBHERecHits_asInput.energyM0.get(),
          HBHERecHits_asInput.timeM0.get(),
          HBHERecHits_asInput.did.get(),
          HBHEPFRecHits_asOutput.PFRecHits_cleaned.pfrh_depth.get(),
          HBHEPFRecHits_asOutput.PFRecHits_cleaned.pfrh_layer.get(),
          HBHEPFRecHits_asOutput.PFRecHits_cleaned.pfrh_detId.get(),
          HBHEPFRecHits_asOutput.PFRecHits_cleaned.pfrh_time.get(),
          HBHEPFRecHits_asOutput.PFRecHits_cleaned.pfrh_energy.get(),
          HBHEPFRecHits_asOutput.PFRecHits_cleaned.pfrh_x.get(),
          HBHEPFRecHits_asOutput.PFRecHits_cleaned.pfrh_y.get(),
          HBHEPFRecHits_asOutput.PFRecHits_cleaned.pfrh_z.get(),
          HBHEPFRecHits_asOutput.PFRecHits_cleaned.pfrh_neighbours.get(),
          false);
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
