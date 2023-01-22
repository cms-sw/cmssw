#ifndef RecoParticleFlow_PFClusterProducer_plugins_DeclsForKernels_h
#define RecoParticleFlow_PFClusterProducer_plugins_DeclsForKernels_h

#include <functional>
#include <optional>

#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHERecHitParamsGPU.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHETopologyGPU.h"

namespace PFRecHit {
  namespace HCAL {

    struct OutputPFRecHitDataGPU {
      ::hcal::PFRecHitCollection<::pf::common::DevStoragePolicy> PFRecHits;

      void allocate(size_t Num_rechits, cudaStream_t cudaStream) {
        PFRecHits.pfrh_depth = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_layer = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_detId = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_neighbours = cms::cuda::make_device_unique<int[]>(Num_rechits * 8, cudaStream);
        PFRecHits.pfrh_neighbourInfos = cms::cuda::make_device_unique<short[]>(Num_rechits * 8, cudaStream);

        PFRecHits.pfrh_time = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_energy = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
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
        rh_pos = cms::cuda::make_host_unique<float3[]>(sizeof(float3) * length, cudaStream);
        rh_detId = cms::cuda::make_host_unique<uint32_t[]>(sizeof(uint32_t) * length, cudaStream);
        rh_neighbours = cms::cuda::make_host_unique<int[]>(sizeof(int) * length * 8, cudaStream);
      }
    };

    struct ScratchDataGPU {
      uint32_t maxSize;
      //cms::cuda::device::unique_ptr<bool[]> rh_mask;
      cms::cuda::device::unique_ptr<int[]> rh_mask;
      cms::cuda::device::unique_ptr<int[]>
          rh_inputToFullIdx;  // Used to build map from input rechit index to lookup table index
      cms::cuda::device::unique_ptr<int[]>
          rh_fullToInputIdx;  // Used to build map from lookup table index to input rechit index
      cms::cuda::device::unique_ptr<int[]>
          pfrhToInputIdx;  // Map PFRecHit index to input rechit index (to account for rechits cut in quality tests)
      cms::cuda::device::unique_ptr<int[]> inputToPFRHIdx;  // Map input rechit index to PF rechit index

      void allocate(uint32_t length, cudaStream_t cudaStream) {
        maxSize = length;
        //rh_mask = cms::cuda::make_device_unique<bool[]>(sizeof(bool)*length, cudaStream);
        rh_mask = cms::cuda::make_device_unique<int[]>(sizeof(int) * length, cudaStream);
        rh_inputToFullIdx = cms::cuda::make_device_unique<int[]>(sizeof(int) * length, cudaStream);
        rh_fullToInputIdx = cms::cuda::make_device_unique<int[]>(sizeof(int) * length, cudaStream);
        pfrhToInputIdx = cms::cuda::make_device_unique<int[]>(sizeof(int) * length, cudaStream);
        inputToPFRHIdx = cms::cuda::make_device_unique<int[]>(sizeof(int) * length, cudaStream);
      }
    };

    // Store EventSetup variables
    struct ConstantProducts {
      PFHBHERecHitParamsGPU::Product const& recHitParametersProduct;
      std::vector<int, cms::cuda::HostAllocator<int>> const& depthHB;
      std::vector<int, cms::cuda::HostAllocator<int>> const& depthHE;
      std::vector<float, cms::cuda::HostAllocator<float>> const& thresholdE_HB;
      std::vector<float, cms::cuda::HostAllocator<float>> const& thresholdE_HE;
      PFHBHETopologyGPU::Product const& topoDataProduct;
      std::vector<uint, cms::cuda::HostAllocator<uint32_t>> const& denseId;
      std::vector<uint, cms::cuda::HostAllocator<uint32_t>> const& detId;
      std::vector<float3, cms::cuda::HostAllocator<float3>> const& position;
      std::vector<int, cms::cuda::HostAllocator<int>> const& neighbours;
    };

  }  // namespace HCAL

  namespace ECAL {

    struct OutputPFRecHitDataGPU {
      ::ecal::PFRecHitCollection<::pf::common::DevStoragePolicy> PFRecHits;

      void allocate(size_t Num_rechits, cudaStream_t cudaStream) {
        PFRecHits.pfrh_depth = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_layer = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_detId = cms::cuda::make_device_unique<int[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_neighbours = cms::cuda::make_device_unique<int[]>(Num_rechits * 8, cudaStream);
        PFRecHits.pfrh_neighbourInfos = cms::cuda::make_device_unique<short[]>(Num_rechits * 8, cudaStream);

        PFRecHits.pfrh_time = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
        PFRecHits.pfrh_energy = cms::cuda::make_device_unique<float[]>(Num_rechits, cudaStream);
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
        rh_pos = cms::cuda::make_host_unique<float3[]>(sizeof(float3) * length, cudaStream);
        rh_detId = cms::cuda::make_host_unique<uint32_t[]>(sizeof(uint32_t) * length, cudaStream);
        rh_neighbours = cms::cuda::make_host_unique<int[]>(sizeof(int) * length * 8, cudaStream);
      }
    };

    struct PersistentDataGPU {
      cms::cuda::device::unique_ptr<float3[]> rh_pos;
      cms::cuda::device::unique_ptr<uint32_t[]> rh_detId;
      cms::cuda::device::unique_ptr<int[]> rh_neighbours;

      void allocate(uint32_t length, cudaStream_t cudaStream) {
        rh_pos = cms::cuda::make_device_unique<float3[]>(sizeof(float3) * length, cudaStream);
        rh_detId = cms::cuda::make_device_unique<uint32_t[]>(sizeof(uint32_t) * length, cudaStream);
        rh_neighbours = cms::cuda::make_device_unique<int[]>(sizeof(int) * length * 8, cudaStream);
      }
    };

    struct ScratchDataGPU {
      uint32_t maxSize;
      cms::cuda::device::unique_ptr<bool[]> rh_mask;
      cms::cuda::device::unique_ptr<int[]>
          rh_inputToFullIdx;  // Used to build map from input rechit index to lookup table index
      cms::cuda::device::unique_ptr<int[]>
          rh_fullToInputIdx;  // Used to build map from lookup table index to input rechit index
      cms::cuda::device::unique_ptr<int[]>
          pfrhToInputIdx;  // Map PFRecHit index to input rechit index (to account for rechits cut in quality tests)
      cms::cuda::device::unique_ptr<int[]> inputToPFRHIdx;  // Map input rechit index to PF rechit index

      void allocate(uint32_t length, cudaStream_t cudaStream) {
        maxSize = length;
        rh_mask = cms::cuda::make_device_unique<bool[]>(sizeof(bool) * length, cudaStream);
        rh_inputToFullIdx = cms::cuda::make_device_unique<int[]>(sizeof(int) * length, cudaStream);
        rh_fullToInputIdx = cms::cuda::make_device_unique<int[]>(sizeof(int) * length, cudaStream);
        pfrhToInputIdx = cms::cuda::make_device_unique<int[]>(sizeof(int) * length, cudaStream);
        inputToPFRHIdx = cms::cuda::make_device_unique<int[]>(sizeof(int) * length, cudaStream);
      }
    };
  }  // namespace ECAL
}  //  namespace PFRecHit

namespace PFClustering {
  namespace HCAL {
    struct ConfigurationParameters {
      uint32_t maxRH = 4000;          // previously: 2000
      uint32_t maxPFCFracs = 300000;  // previously: 80000
      uint32_t maxNeighbors = 8;
    };

    struct OutputDataCPU {
      cms::cuda::host::unique_ptr<int[]> pfrh_topoId;
      cms::cuda::host::unique_ptr<int[]> pfrh_isSeed;
      cms::cuda::host::unique_ptr<float[]> pcrh_frac;
      cms::cuda::host::unique_ptr<int[]> pcrh_fracInd;
      //cms::cuda::host::unique_ptr<bool[]> pfrh_passTopoThresh;
      cms::cuda::host::unique_ptr<int[]> pfrh_passTopoThresh;

      cms::cuda::host::unique_ptr<int[]> topoSeedCount;
      cms::cuda::host::unique_ptr<int[]> topoRHCount;
      cms::cuda::host::unique_ptr<int[]> seedFracOffsets;
      cms::cuda::host::unique_ptr<int[]> topoSeedOffsets;
      cms::cuda::host::unique_ptr<int[]> topoSeedList;

      cms::cuda::host::unique_ptr<int[]> pfc_iter;  // Iterations per pf cluster by seed index. For debugging use only
      cms::cuda::host::unique_ptr<int[]> topoIter;  // Iterations for topo clustering to converge. For debugging use only
      cms::cuda::host::unique_ptr<int[]> pcrhFracSize;  // Total number of pfc fractions to copy back
      cms::cuda::host::unique_ptr<int[]> nEdges;        // Sum total number of rechit neighbours

      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = cudaStreamDefault) {
        pfrh_topoId = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pfrh_isSeed = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pcrh_frac = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxPFCFracs, cudaStream);
        pcrh_fracInd = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxPFCFracs, cudaStream);
        //pfrh_passTopoThresh = cms::cuda::make_host_unique<bool[]>(sizeof(bool)*config.maxRH, cudaStream);
        pfrh_passTopoThresh = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);

        topoSeedCount = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoRHCount = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        seedFracOffsets = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoSeedOffsets = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoSeedList = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);

        pfc_iter = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoIter = cms::cuda::make_host_unique<int[]>(sizeof(int), cudaStream);
        pcrhFracSize = cms::cuda::make_host_unique<int[]>(sizeof(int), cudaStream);
        nEdges = cms::cuda::make_host_unique<int[]>(sizeof(int), cudaStream);
      }
    };

    struct OutputDataGPU {
      cms::cuda::device::unique_ptr<int[]> pfrh_topoId;
      cms::cuda::device::unique_ptr<int[]> pfrh_isSeed;
      cms::cuda::device::unique_ptr<float[]> pcrh_frac;
      cms::cuda::device::unique_ptr<int[]> pcrh_fracInd;
      //cms::cuda::device::unique_ptr<bool[]> pfrh_passTopoThresh;
      cms::cuda::device::unique_ptr<int[]> pfrh_passTopoThresh;

      cms::cuda::device::unique_ptr<int[]> topoSeedCount;
      cms::cuda::device::unique_ptr<int[]> topoRHCount;
      cms::cuda::device::unique_ptr<int[]> seedFracOffsets;
      cms::cuda::device::unique_ptr<int[]> topoSeedOffsets;
      cms::cuda::device::unique_ptr<int[]> topoSeedList;

      cms::cuda::device::unique_ptr<int[]> pfc_iter;  // Iterations per pf cluster by seed index. For debugging use only
      cms::cuda::device::unique_ptr<int[]>
          topoIter;  // Iterations for topo clustering to converge. For debugging use only
      cms::cuda::device::unique_ptr<int[]> pcrhFracSize;  // Total number of pfc fractions to copy back
      cms::cuda::device::unique_ptr<int[]> nEdges;        // Sum total number of rechit neighbours

      cms::cuda::device::unique_ptr<float4[]> pfc_pos4;
       cms::cuda::device::unique_ptr<float[]> pfc_energy;

      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = cudaStreamDefault) {
        pfrh_topoId = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pfrh_isSeed = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pcrh_frac = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxPFCFracs, cudaStream);
        pcrh_fracInd = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxPFCFracs, cudaStream);
        //pfrh_passTopoThresh = cms::cuda::make_device_unique<bool[]>(sizeof(bool)*config.maxRH, cudaStream);
        pfrh_passTopoThresh = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);

        topoSeedCount = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoRHCount = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        seedFracOffsets = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoSeedOffsets = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoSeedList = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);

        pfc_iter = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoIter = cms::cuda::make_device_unique<int[]>(sizeof(int), cudaStream);
        pcrhFracSize = cms::cuda::make_device_unique<int[]>(sizeof(int), cudaStream);
        nEdges = cms::cuda::make_device_unique<int[]>(sizeof(int), cudaStream);

        pfc_pos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4) * config.maxRH, cudaStream);
        pfc_energy = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
      }
    };

    struct ScratchDataGPU {
      cms::cuda::device::unique_ptr<int[]> rhcount;
      cms::cuda::device::unique_ptr<int[]> pfrh_edgeId;
      cms::cuda::device::unique_ptr<int[]> pfrh_edgeList;
      cms::cuda::device::unique_ptr<int[]> pfrh_edgeMask;

      cms::cuda::device::unique_ptr<float[]> pcrh_fracSum;
      cms::cuda::device::unique_ptr<float4[]> pfc_prevPos4;

      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = cudaStreamDefault) {
        rhcount = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pfrh_edgeId =
            cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);
        pfrh_edgeList =
            cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);
        pfrh_edgeMask =
            cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);

        pcrh_fracSum = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfc_prevPos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4) * config.maxRH, cudaStream);

      }
    };
  }  // namespace HCAL

}  // namespace PFClustering

#endif  // RecoParticleFlow_PFClusterProducer_plugins_DeclsForKernels_h
