#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Uncomment to enable GPU debugging
//#define DEBUG_GPU_ECAL

// Uncomment to fill TTrees
//#define DEBUG_ECAL_TREES

// Uncomment to save cluster collections in TTree
//#define DEBUG_SAVE_CLUSTERS

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>

// CMSSW include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

#include "CudaPFCommon.h"
#include "PFClusterCudaECAL.h"

namespace PFClustering {
  namespace ECAL {
    struct ConfigurationParameters {
      uint32_t maxRH = 2000;
      uint32_t maxPFCSize = 75;
      uint32_t maxPFCFracs = 80000;
      uint32_t maxNeighbors = 8;
    };

    struct InputDataCPU {
      cms::cuda::host::unique_ptr<float[]> pfrh_x;
      cms::cuda::host::unique_ptr<float[]> pfrh_y;
      cms::cuda::host::unique_ptr<float[]> pfrh_z;
      cms::cuda::host::unique_ptr<float[]> pfrh_energy;
      cms::cuda::host::unique_ptr<float[]> pfrh_pt2;

      // m_axis
      cms::cuda::host::unique_ptr<float[]> rh_axis_x;
      cms::cuda::host::unique_ptr<float[]> rh_axis_y;
      cms::cuda::host::unique_ptr<float[]> rh_axis_z;

      cms::cuda::host::unique_ptr<int[]> pfrh_layer;
      cms::cuda::host::unique_ptr<int[]> pfNeighEightInd;
      cms::cuda::host::unique_ptr<int[]> pfrh_edgeId;
      cms::cuda::host::unique_ptr<int[]> pfrh_edgeList;

      void allocate(ConfigurationParameters const &config, cudaStream_t cudaStream = cudaStreamDefault) {
        pfrh_x = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfrh_y = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfrh_z = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfrh_energy = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfrh_pt2 = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);

        // Detector geometry
        rh_axis_x = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        rh_axis_y = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        rh_axis_z = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);

        pfrh_layer = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pfNeighEightInd =
            cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);
        pfrh_edgeId = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);
        pfrh_edgeList =
            cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);
      }
    };

    struct OutputDataCPU {
      cms::cuda::host::unique_ptr<int[]> pfrh_topoId;
      cms::cuda::host::unique_ptr<int[]> pfrh_isSeed;
      cms::cuda::host::unique_ptr<float[]> pcrh_frac;
      cms::cuda::host::unique_ptr<int[]> pcrh_fracInd;
      cms::cuda::host::unique_ptr<bool[]> pfrh_passTopoThresh;

      cms::cuda::host::unique_ptr<int[]> topoSeedCount;
      cms::cuda::host::unique_ptr<int[]> topoRHCount;
      cms::cuda::host::unique_ptr<int[]> seedFracOffsets;
      cms::cuda::host::unique_ptr<int[]> topoSeedOffsets;
      cms::cuda::host::unique_ptr<int[]> topoSeedList;

      cms::cuda::host::unique_ptr<int[]> pfc_iter;      // Iterations per pf cluster (by seed index)
      cms::cuda::host::unique_ptr<int[]> topoIter;      // Iterations for topo clustering to converge
      cms::cuda::host::unique_ptr<int[]> pcrhFracSize;  // Total number of pfc fractions to copy back

      void allocate(ConfigurationParameters const &config, cudaStream_t cudaStream = cudaStreamDefault) {
        pfrh_topoId = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pfrh_isSeed = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pcrh_frac = cms::cuda::make_host_unique<float[]>(sizeof(float) * config.maxPFCFracs, cudaStream);
        pcrh_fracInd = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxPFCFracs, cudaStream);
        pfrh_passTopoThresh = cms::cuda::make_host_unique<bool[]>(sizeof(bool) * config.maxRH, cudaStream);

        topoSeedCount = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoRHCount = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        seedFracOffsets = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoSeedOffsets = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoSeedList = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);

        pfc_iter = cms::cuda::make_host_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoIter = cms::cuda::make_host_unique<int[]>(sizeof(int), cudaStream);
        pcrhFracSize = cms::cuda::make_host_unique<int[]>(sizeof(int), cudaStream);
      }
    };

    struct InputDataGPU {
      cms::cuda::device::unique_ptr<float[]> pfrh_x;
      cms::cuda::device::unique_ptr<float[]> pfrh_y;
      cms::cuda::device::unique_ptr<float[]> pfrh_z;
      cms::cuda::device::unique_ptr<float[]> pfrh_energy;
      cms::cuda::device::unique_ptr<float[]> pfrh_pt2;

      // m_axis
      cms::cuda::device::unique_ptr<float[]> rh_axis_x;
      cms::cuda::device::unique_ptr<float[]> rh_axis_y;
      cms::cuda::device::unique_ptr<float[]> rh_axis_z;

      cms::cuda::device::unique_ptr<int[]> pfrh_layer;
      cms::cuda::device::unique_ptr<int[]> pfNeighEightInd;
      cms::cuda::device::unique_ptr<int[]> pfrh_edgeId;
      cms::cuda::device::unique_ptr<int[]> pfrh_edgeList;
      cms::cuda::device::unique_ptr<int[]> pfrh_edgeMask;

      cms::cuda::device::unique_ptr<bool[]> pfrh_passTopoThresh;

      void allocate(ConfigurationParameters const &config, cudaStream_t cudaStream = cudaStreamDefault) {
        pfrh_x = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfrh_y = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfrh_z = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfrh_energy = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfrh_pt2 = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);

        // Detector geometry
        rh_axis_x = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        rh_axis_y = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        rh_axis_z = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);

        pfrh_layer = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pfNeighEightInd =
            cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);
        pfrh_edgeId =
            cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);
        pfrh_edgeList =
            cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);
        pfrh_edgeMask =
            cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH * config.maxNeighbors, cudaStream);
        pfrh_passTopoThresh = cms::cuda::make_device_unique<bool[]>(sizeof(bool) * config.maxRH, cudaStream);
      }
    };

    struct OutputDataGPU {
      cms::cuda::device::unique_ptr<int[]> pfrh_topoId;
      cms::cuda::device::unique_ptr<int[]> pfrh_isSeed;
      cms::cuda::device::unique_ptr<float[]> pcrh_frac;
      cms::cuda::device::unique_ptr<int[]> pcrh_fracInd;
      cms::cuda::device::unique_ptr<bool[]> pfrh_passTopoThresh;

      cms::cuda::device::unique_ptr<int[]> topoSeedCount;
      cms::cuda::device::unique_ptr<int[]> topoRHCount;
      cms::cuda::device::unique_ptr<int[]> seedFracOffsets;
      cms::cuda::device::unique_ptr<int[]> topoSeedOffsets;
      cms::cuda::device::unique_ptr<int[]> topoSeedList;

      cms::cuda::device::unique_ptr<int[]> pfc_iter;      // Iterations per pf cluster (by seed index)
      cms::cuda::device::unique_ptr<int[]> topoIter;      // Iterations for topo clustering to converge
      cms::cuda::device::unique_ptr<int[]> pcrhFracSize;  // Total number of pfc fractions to copy back

      void allocate(ConfigurationParameters const &config, cudaStream_t cudaStream = cudaStreamDefault) {
        pfrh_topoId = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pfrh_isSeed = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pcrh_frac = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxPFCFracs, cudaStream);
        pcrh_fracInd = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxPFCFracs, cudaStream);
        pfrh_passTopoThresh = cms::cuda::make_device_unique<bool[]>(sizeof(bool) * config.maxRH, cudaStream);

        topoSeedCount = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoRHCount = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        seedFracOffsets = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoSeedOffsets = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoSeedList = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);

        pfc_iter = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        topoIter = cms::cuda::make_device_unique<int[]>(sizeof(int), cudaStream);
        pcrhFracSize = cms::cuda::make_device_unique<int[]>(sizeof(int), cudaStream);
      }
    };

    struct ScratchDataGPU {
      cms::cuda::device::unique_ptr<float4[]> pfc_convPos4;    // Convergence position
      cms::cuda::device::unique_ptr<float4[]> pfc_prevPos4;    // Previous (convergence) position
      cms::cuda::device::unique_ptr<float4[]> pfc_linearPos4;  // Linear position
      cms::cuda::device::unique_ptr<float4[]> pfc_pos4;        // Cluster position
      cms::cuda::device::unique_ptr<float[]> pfc_energy;       // Cluster energy
      cms::cuda::device::unique_ptr<float[]> pfc_clusterT0;    // Cluster T0 (for 2D position calculation)

      cms::cuda::device::unique_ptr<int[]> rhcount;
      cms::cuda::device::unique_ptr<float[]> pcrh_fracSum;

      void allocate(ConfigurationParameters const &config,
                    cudaStream_t cudaStream = cudaStreamDefault /* default Cuda stream */) {
        pfc_convPos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4) * config.maxRH, cudaStream);
        pfc_prevPos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4) * config.maxRH, cudaStream);
        pfc_linearPos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4) * config.maxRH, cudaStream);
        pfc_pos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4) * config.maxRH, cudaStream);
        pfc_energy = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
        pfc_clusterT0 = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);

        rhcount = cms::cuda::make_device_unique<int[]>(sizeof(int) * config.maxRH, cudaStream);
        pcrh_fracSum = cms::cuda::make_device_unique<float[]>(sizeof(float) * config.maxRH, cudaStream);
      }
    };

  }  // namespace ECAL
}  // namespace PFClustering

class PFClusterProducerCudaECAL : public edm::stream::EDProducer<edm::ExternalWork> {
  typedef RecHitTopologicalCleanerBase RHCB;
  typedef InitialClusteringStepBase ICSB;
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;

public:
  PFClusterProducerCudaECAL(const edm::ParameterSet &);
  ~PFClusterProducerCudaECAL() override;

  // options
  const bool _prodInitClusters;
  // the actual algorithm
  std::vector<std::unique_ptr<RecHitTopologicalCleanerBase>> _cleaners;
  std::unique_ptr<SeedFinderBase> _seedFinder;
  std::unique_ptr<InitialClusteringStepBase> _initialClustering;
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFCPositionCalculatorBase> _positionReCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _convergencePosCalc;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;

  TFile *MyFile = new TFile("EventECAL.root", "recreate");

  reco::PFClusterCollection __initialClusters;
  reco::PFClusterCollection __pfClusters;
  reco::PFClusterCollection __pfClustersFromCuda;
  reco::PFRecHitCollection __rechits;

  // rechit physics quantities
  std::vector<int> __rh_mask;
  std::vector<int> __rh_isSeed;
  std::vector<float> __rh_x;
  std::vector<float> __rh_y;
  std::vector<float> __rh_z;
  std::vector<float> __rh_axis_x;
  std::vector<float> __rh_axis_y;
  std::vector<float> __rh_axis_z;
  std::vector<float> __rh_eta;
  std::vector<float> __rh_phi;
  std::vector<float> __rh_pt2;
  // rechit neighbours4, neighbours8 vectors
  std::vector<std::vector<unsigned int>> __rh_neighbours8;

  std::vector<int> __pfcIter;
  std::vector<int> __nRHTopo;
  std::vector<int> __nSeedsTopo;
  std::vector<int> __nFracsTopo;

  TTree *clusterTree = new TTree("clusterTree", "clusterTree");

  TH1F *pfcIterations = new TH1F("pfcIter", "nIterations PF Clustering", 50, 0.5, 50.5);
  TH2F *pfcIter_vs_nRHTopo =
      new TH2F("pfcIternRHTopo", "nIterations PF Clustering vs nRH topo", 300, 0.5, 300.5, 50, 0.5, 50.5);
  TH2F *pfcIter_vs_nSeedsTopo =
      new TH2F("pfcIternSeedsTopo", "nIterations PF Clustering vs nSeeds in topo", 75, 0.5, 75.5, 50, 0.5, 50.5);
  TH2F *pfcIter_vs_nFracsTopo =
      new TH2F("pfcIternFracsTopo", "nIterations PF Clustering vs nFracs in topo", 260, 0.5, 260.5, 50, 0.5, 50.5);

  TH2F *nFracs_vs_nRH = new TH2F("nFracsnRH", "PF Cluster Fractions vs num rechits", 50, 0.5, 2000.5, 50, 0.5, 70000.5);

  TH1F *topoIterations = new TH1F("topoIter", "nIterations Topo Clustering", 25, 0.5, 25.5);
  TH2F *topoIter_vs_nRH =
      new TH2F("topoIternRH", "nIterations Topo Clustering vs num rechits", 3000, 0.5, 3000.5, 25, 0.5, 25.5);
  TH1F *nTopo_CPU = new TH1F("nTopo_CPU", "nTopo_CPU", 500, 0.5, 500.5);
  TH1F *nTopo_GPU = new TH1F("nTopo_GPU", "nTopo_GPU", 500, 0.5, 500.5);

  TH1F *topoSeeds_CPU = new TH1F("topoSeeds_CPU", "topoSeeds_CPU", 200, 0.5, 200.5);
  TH1F *topoSeeds_GPU = new TH1F("topoSeeds_GPU", "topoSeeds_GPU", 200, 0.5, 200.5);

  TH1F *sumSeed_CPU = new TH1F("sumSeed_CPU", "sumSeed_CPU", 200, 0.5, 200.5);
  TH1F *sumSeed_GPU = new TH1F("sumSeed_GPU", "sumSeed_GPU", 200, 0.5, 200.5);

  TH1F *topoEn_CPU = new TH1F("topoEn_CPU", "topoEn_CPU", 500, 0, 500);
  TH1F *topoEn_GPU = new TH1F("topoEn_GPU", "topoEn_GPU", 500, 0, 500);

  TH1F *topoEta_CPU = new TH1F("topoEta_CPU", "topoEta_CPU", 100, -3, 3);
  TH1F *topoEta_GPU = new TH1F("topoEta_GPU", "topoEta_GPU", 100, -3, 3);

  TH1F *topoPhi_CPU = new TH1F("topoPhi_CPU", "topoPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *topoPhi_GPU = new TH1F("topoPhi_GPU", "topoPhi_GPU", 100, -3.1415926, 3.1415926);

  TH1F *nPFCluster_CPU = new TH1F("nPFCluster_CPU", "nPFCluster_CPU", 1000, 0.5, 1000.5);
  TH1F *nPFCluster_GPU = new TH1F("nPFCluster_GPU", "nPFCluster_GPU", 1000, 0.5, 1000.5);

  TH1F *enPFCluster_CPU = new TH1F("enPFCluster_CPU", "enPFCluster_CPU", 500, 0, 500);
  TH1F *enPFCluster_GPU = new TH1F("enPFCluster_GPU", "enPFCluster_GPU", 500, 0, 500);

  TH1F *pfcEta_CPU = new TH1F("pfcEta_CPU", "pfcEta_CPU", 100, -3, 3);
  TH1F *pfcEta_GPU = new TH1F("pfcEta_GPU", "pfcEta_GPU", 100, -3, 3);

  TH1F *pfcPhi_CPU = new TH1F("pfcPhi_CPU", "pfcPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *pfcPhi_GPU = new TH1F("pfcPhi_GPU", "pfcPhi_GPU", 100, -3.1415926, 3.1415926);

  TH1F *nRH_perPFCluster_CPU = new TH1F("nRH_perPFCluster_CPU", "nRH_perPFCluster_CPU", 101, -0.5, 100.5);
  TH1F *nRH_perPFCluster_GPU = new TH1F("nRH_perPFCluster_GPU", "nRH_perPFCluster_GPU", 101, -0.5, 100.5);

  // Total number of rechit fractions in all PF clusters per event (includes float counting)
  TH1F *nRH_perPFClusterTotal_CPU =
      new TH1F("nRH_perPFClusterTotal_CPU", "nRH_perPFClusterTotal_CPU", 2000, 0.5, 2000.5);
  TH1F *nRH_perPFClusterTotal_GPU =
      new TH1F("nRH_perPFClusterTotal_GPU", "nRH_perPFClusterTotal_GPU", 2000, 0.5, 2000.5);

  TH1F *matched_pfcRh_CPU = new TH1F("matched_pfcRh_CPU", "matching seed pfcRh_CPU", 101, -0.5, 100.5);
  TH1F *matched_pfcRh_GPU = new TH1F("matched_pfcRh_GPU", "matching seed pfcRh_GPU", 101, -0.5, 100.5);

  TH1F *matched_pfcEn_CPU = new TH1F("matched_pfcEn_CPU", "matching seed pfcEn_CPU", 500, 0, 500);
  TH1F *matched_pfcEn_GPU = new TH1F("matched_pfcEn_GPU", "matching seed pfcEn_GPU", 500, 0, 500);

  TH1F *matched_pfcEta_CPU = new TH1F("matched_pfcEta_CPU", "matching seed pfcEta_CPU", 100, -3, 3);
  TH1F *matched_pfcEta_GPU = new TH1F("matched_pfcEta_GPU", "matching seed pfcEta_GPU", 100, -3, 3);

  TH1F *matched_pfcPhi_CPU = new TH1F("matched_pfcPhi_CPU", "matching seed pfcPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *matched_pfcPhi_GPU = new TH1F("matched_pfcPhi_GPU", "matching seed pfcPhi_GPU", 100, -3.1415926, 3.1415926);

  TH2F *nRh_CPUvsGPU = new TH2F("nRh_CPUvsGPU", "nRh_CPUvsGPU", 101, -0.5, 100.5, 101, -0.5, 100.5);
  TH2F *enPFCluster_CPUvsGPU = new TH2F("enPFCluster_CPUvsGPU", "enPFCluster_CPUvsGPU", 50, 0, 500, 50, 0, 500);

  TH1F *deltaSumSeed = new TH1F("deltaSumSeed", "sumSeed_{GPU} - sumSeed_{CPU}", 201, -100.5, 100.5);
  TH1F *deltaRH = new TH1F("deltaRH", "nRH_{GPU} - nRH_{CPU}", 41, -20.5, 20.5);
  TH1F *deltaEn = new TH1F("deltaEn", "E_{GPU} - E_{CPU}", 200, -10, 10);
  TH1F *deltaEta = new TH1F("deltaEta", "#eta_{GPU} - #eta_{CPU}", 200, -0.2, 0.2);
  TH1F *deltaPhi = new TH1F("deltaPhi", "#phi_{GPU} - #phi_{CPU}", 200, -0.2, 0.2);

  TH2F *coordinate = new TH2F("coordinate", "coordinate", 100, -3, 3, 100, -3.1415926, 3.14159);
  TH1F *layer = new TH1F("layer", "layer", 7, 0, 7);

  TH1F *hTimers = new TH1F("timers", "GPU kernel timers (Event > 9)", 9, -0.5, 8.5);
  std::array<float, 9> GPU_timers;
  Int_t numEvents = 0;
  Int_t topoIter = 0;
  Int_t nEdges = 0;
  Int_t nFracs = 0;

  Int_t nRHperPFCTotal_CPU = 0;
  Int_t nRHperPFCTotal_GPU = 0;

private:
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void acquire(edm::Event const &, edm::EventSetup const &, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event &, const edm::EventSetup &) override;

  //bool doComparison=true;
  bool doComparison = false;

  // inputs
  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;

  cms::cuda::ContextState cudaState_;
  cudaStream_t cudaStream = cudaStreamDefault;

  PFClustering::common::CudaECALConstants cudaConstants;
  PFClustering::ECAL::ConfigurationParameters cudaConfig_;
  PFClustering::ECAL::InputDataCPU inputCPU;
  PFClustering::ECAL::OutputDataCPU outputCPU;
  PFClustering::ECAL::InputDataGPU inputGPU;
  PFClustering::ECAL::OutputDataGPU outputGPU;
  PFClustering::ECAL::ScratchDataGPU scratchGPU;

  std::unique_ptr<reco::PFClusterCollection> pfClustersFromCuda;
};

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

PFClusterProducerCudaECAL::PFClusterProducerCudaECAL(const edm::ParameterSet &conf)
    : _prodInitClusters(conf.getUntrackedParameter<bool>("prodInitialClusters", false)) {
  _rechitsLabel = consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource"));
  edm::ConsumesCollector cc = consumesCollector();

  nFracs_vs_nRH->GetXaxis()->SetTitle("nRH");
  nFracs_vs_nRH->GetYaxis()->SetTitle("nFracs");

  pfcIterations->GetXaxis()->SetTitle("PF clustering iterations");
  pfcIterations->GetYaxis()->SetTitle("Entries");

  pfcIter_vs_nRHTopo->GetXaxis()->SetTitle("Num rechits in topo cluster");
  pfcIter_vs_nRHTopo->GetYaxis()->SetTitle("PF clustering iterations");

  pfcIter_vs_nSeedsTopo->GetXaxis()->SetTitle("Num seeds in topo cluster");
  pfcIter_vs_nSeedsTopo->GetYaxis()->SetTitle("PF clustering iterations");

  pfcIter_vs_nFracsTopo->GetXaxis()->SetTitle("Num rechit fractions in topo cluster");
  pfcIter_vs_nFracsTopo->GetYaxis()->SetTitle("PF clustering iterations");

  topoIterations->GetXaxis()->SetTitle("Topo clustering iterations");
  topoIterations->GetYaxis()->SetTitle("Entries");

  topoIter_vs_nRH->GetXaxis()->SetTitle("Num rechits");
  topoIter_vs_nRH->GetYaxis()->SetTitle("Topo clustering iterations");

#ifdef DEBUG_ECAL_TREES
  hTimers->GetYaxis()->SetTitle("time (ms)");
  hTimers->GetXaxis()->SetBinLabel(1, "copyToDevice");
  hTimers->GetXaxis()->SetBinLabel(2, "seeding");
  hTimers->GetXaxis()->SetBinLabel(3, "topo");
  hTimers->GetXaxis()->SetBinLabel(4, "setup frac");
  hTimers->GetXaxis()->SetBinLabel(5, "PF clustering");
  hTimers->GetXaxis()->SetBinLabel(6, "copyToHost");

  //setup TTree
  clusterTree->Branch("Event", &numEvents);
  clusterTree->Branch("topoIter", &topoIter, "topoIter/I");
  clusterTree->Branch("nEdges", &nEdges, "nEdges/I");
  clusterTree->Branch("nFracs", &nFracs, "nFracs/I");
  clusterTree->Branch("nRHperPFCTotal_CPU", &nRHperPFCTotal_CPU, "nRHperPFCTotal_CPU/I");
  clusterTree->Branch("nRHperPFCTotal_GPU", &nRHperPFCTotal_GPU, "nRHperPFCTotal_GPU/I");
  clusterTree->Branch("timers", &GPU_timers);
  clusterTree->Branch("pfcIter", &__pfcIter);
  clusterTree->Branch("nRHTopo", &__nRHTopo);
  clusterTree->Branch("nSeedsTopo", &__nSeedsTopo);
  clusterTree->Branch("nFracsTopo", &__nFracsTopo);
  clusterTree->Branch("rechits", "PFRecHitCollection", &__rechits);
  clusterTree->Branch("rechits_mask", &__rh_mask);
  clusterTree->Branch("rechits_isSeed", &__rh_isSeed);
  clusterTree->Branch("rechits_x", &__rh_x);
  clusterTree->Branch("rechits_y", &__rh_y);
  clusterTree->Branch("rechits_z", &__rh_z);
  clusterTree->Branch("rechits_eta", &__rh_eta);
  clusterTree->Branch("rechits_phi", &__rh_phi);
  clusterTree->Branch("rechits_pt2", &__rh_pt2);
  clusterTree->Branch("rechits_neighbours8", &__rh_neighbours8);
  clusterTree->Branch("rechits_rh_axis_x", &__rh_axis_x);
  clusterTree->Branch("rechits_rh_axis_y", &__rh_axis_y);
  clusterTree->Branch("rechits_rh_axis_z", &__rh_axis_z);
#endif
#if defined DEBUG_ECAL_TREES && defined DEBUG_SAVE_CLUSTERS
  //setup TTree
  clusterTree->Branch("initialClusters", "PFClusterCollection", &__initialClusters);
  clusterTree->Branch("pfClusters", "PFClusterCollection", &__pfClusters);
  clusterTree->Branch("pfClustersFromCuda", "PFClusterCollection", &__pfClustersFromCuda);
#endif

  //setup rechit cleaners
  const edm::VParameterSet &cleanerConfs = conf.getParameterSetVector("recHitCleaners");

  for (const auto &conf : cleanerConfs) {
    const std::string &cleanerName = conf.getParameter<std::string>("algoName");
    _cleaners.emplace_back(RecHitTopologicalCleanerFactory::get()->create(cleanerName, conf, cc));
  }

  // setup seed finding
  const edm::ParameterSet &sfConf = conf.getParameterSet("seedFinder");
  const std::string &sfName = sfConf.getParameter<std::string>("algoName");
  _seedFinder = SeedFinderFactory::get()->create(sfName, sfConf);

  //setup topo cluster builder
  const edm::ParameterSet &initConf = conf.getParameterSet("initialClusteringStep");
  const std::string &initName = initConf.getParameter<std::string>("algoName");
  _initialClustering = InitialClusteringStepFactory::get()->create(initName, initConf, cc);
  //setup pf cluster builder if requested
  const edm::ParameterSet &pfcConf = conf.getParameterSet("pfClusterBuilder");
  if (!pfcConf.empty()) {
    const std::string &pfcName = pfcConf.getParameter<std::string>("algoName");
    _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf, cc);
  }

  if (pfcConf.exists("allCellsPositionCalc")) {
    const edm::ParameterSet &acConf = pfcConf.getParameterSet("allCellsPositionCalc");
    const std::string &algoac = acConf.getParameter<std::string>("algoName");
    _allCellsPosCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
    cudaConstants.posCalcConfig.minAllowedNormalization = (float)acConf.getParameter<double>("minAllowedNormalization");
    cudaConstants.posCalcConfig.logWeightDenominatorInv =
        1. / (float)acConf.getParameter<double>("logWeightDenominator");
    cudaConstants.posCalcConfig.minFractionInCalc = (float)acConf.getParameter<double>("minFractionInCalc");
  }

  if (pfcConf.exists("positionCalcForConvergence")) {
    const edm::ParameterSet &convConf = pfcConf.getParameterSet("positionCalcForConvergence");
    if (!convConf.empty()) {
      const std::string &pName = convConf.getParameter<std::string>("algoName");
      _convergencePosCalc = PFCPositionCalculatorFactory::get()->create(pName, convConf, cc);
      cudaConstants.convergencePosCalcConfig.minAllowedNormalization =
          (float)convConf.getParameter<double>("minAllowedNormalization");
      cudaConstants.convergencePosCalcConfig.T0_ES = (float)convConf.getParameter<double>("T0_ES");
      cudaConstants.convergencePosCalcConfig.T0_EE = (float)convConf.getParameter<double>("T0_EE");
      cudaConstants.convergencePosCalcConfig.T0_EB = (float)convConf.getParameter<double>("T0_EB");
      cudaConstants.convergencePosCalcConfig.X0 = (float)convConf.getParameter<double>("X0");
      cudaConstants.convergencePosCalcConfig.minFractionInCalc =
          (float)convConf.getParameter<double>("minFractionInCalc");
      cudaConstants.convergencePosCalcConfig.W0 = (float)convConf.getParameter<double>("W0");
    }
  }

  //setup (possible) recalcuation of positions
  const edm::ParameterSet &pConf = conf.getParameterSet("positionReCalc");
  if (!pConf.empty()) {
    const std::string &pName = pConf.getParameter<std::string>("algoName");
    _positionReCalc = PFCPositionCalculatorFactory::get()->create(pName, pConf, cc);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet &cConf = conf.getParameterSet("energyCorrector");
  if (!cConf.empty()) {
    const std::string &cName = cConf.getParameter<std::string>("algoName");
    _energyCorrector = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
  }

  // Initialize Cuda device constant values
  // Read values from parameter set
  cudaConstants.showerSigma2 = (float)std::pow(pfcConf.getParameter<double>("showerSigma"), 2.);
  const auto &recHitEnergyNormConf = pfcConf.getParameterSetVector("recHitEnergyNorms");
  for (const auto &pset : recHitEnergyNormConf) {
    const std::string &det = pset.getParameter<std::string>("detector");
    if (det == std::string("ECAL_BARREL"))
      cudaConstants.recHitEnergyNormInvEB = (float)(1. / pset.getParameter<double>("recHitEnergyNorm"));
    else if (det == std::string("ECAL_ENDCAP"))
      cudaConstants.recHitEnergyNormInvEE = (float)(1. / pset.getParameter<double>("recHitEnergyNorm"));
    else
      std::cout << "Unknown detector when parsing recHitEnergyNorm: " << det << std::endl;
  }

  cudaConstants.minFracToKeep = (float)pfcConf.getParameter<double>("minFractionToKeep");
  cudaConstants.minFracTot = (float)pfcConf.getParameter<double>("minFracTot");
  cudaConstants.maxIterations = pfcConf.getParameter<unsigned>("maxIterations");
  cudaConstants.excludeOtherSeeds = pfcConf.getParameter<bool>("excludeOtherSeeds");
  cudaConstants.stoppingTolerance = (float)pfcConf.getParameter<double>("stoppingTolerance");

  const auto &seedThresholdConf = sfConf.getParameterSetVector("thresholdsByDetector");
  for (const auto &pset : seedThresholdConf) {
    const std::string &det = pset.getParameter<std::string>("detector");
    if (det == std::string("ECAL_BARREL")) {
      cudaConstants.seedEThresholdEB = (float)pset.getParameter<double>("seedingThreshold");
      cudaConstants.seedPt2ThresholdEB = (float)std::pow(pset.getParameter<double>("seedingThresholdPt"), 2.);
    } else if (det == std::string("ECAL_ENDCAP")) {
      cudaConstants.seedEThresholdEE = (float)pset.getParameter<double>("seedingThreshold");
      cudaConstants.seedPt2ThresholdEE = (float)std::pow(pset.getParameter<double>("seedingThresholdPt"), 2.);
    } else
      std::cout << "Unknown detector when parsing seedFinder: " << det << std::endl;
  }

  const auto &topoThresholdConf = initConf.getParameterSetVector("thresholdsByDetector");
  for (const auto &pset : topoThresholdConf) {
    const std::string &det = pset.getParameter<std::string>("detector");
    if (det == std::string("ECAL_BARREL")) {
      cudaConstants.topoEThresholdEB = (float)pset.getParameter<double>("gatheringThreshold");
    } else if (det == std::string("ECAL_ENDCAP")) {
      cudaConstants.topoEThresholdEE = (float)pset.getParameter<double>("gatheringThreshold");
    } else
      std::cout << "Unknown detector when parsing initClusteringStep: " << det << std::endl;
  }
  cudaConstants.nNeigh = sfConf.getParameter<int>("nNeighbours");

  pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();

  if (_prodInitClusters) {
    produces<reco::PFClusterCollection>("initialClusters");
  }
  produces<reco::PFClusterCollection>();
}

PFClusterProducerCudaECAL::~PFClusterProducerCudaECAL() {
  MyFile->cd();
#ifdef DEBUG_ECAL_TREES
  clusterTree->Write();
#endif
  nFracs_vs_nRH->Write();
  pfcIterations->Write();
  pfcIter_vs_nRHTopo->Write();
  pfcIter_vs_nSeedsTopo->Write();
  pfcIter_vs_nFracsTopo->Write();
  topoIterations->Write();
  topoIter_vs_nRH->Write();
  nTopo_CPU->Write();
  nTopo_GPU->Write();
  topoSeeds_CPU->Write();
  topoSeeds_GPU->Write();
  sumSeed_CPU->Write();
  sumSeed_GPU->Write();
  topoEn_CPU->Write();
  topoEn_GPU->Write();
  topoEta_CPU->Write();
  topoEta_GPU->Write();
  topoPhi_CPU->Write();
  topoPhi_GPU->Write();
  nPFCluster_CPU->Write();
  nPFCluster_GPU->Write();
  enPFCluster_CPU->Write();
  enPFCluster_GPU->Write();
  pfcEta_CPU->Write();
  pfcEta_GPU->Write();
  pfcPhi_CPU->Write();
  pfcPhi_GPU->Write();
  nRH_perPFCluster_CPU->Write();
  nRH_perPFCluster_GPU->Write();
  nRH_perPFClusterTotal_CPU->Write();
  nRH_perPFClusterTotal_GPU->Write();
  matched_pfcRh_CPU->Write();
  matched_pfcRh_GPU->Write();
  matched_pfcEn_CPU->Write();
  matched_pfcEn_GPU->Write();
  matched_pfcEta_CPU->Write();
  matched_pfcEta_GPU->Write();
  matched_pfcPhi_CPU->Write();
  matched_pfcPhi_GPU->Write();
  nRh_CPUvsGPU->Write();
  enPFCluster_CPUvsGPU->Write();
  coordinate->Write();
  layer->Write();
  deltaSumSeed->Write();
  deltaRH->Write();
  deltaEn->Write();
  deltaEta->Write();
  deltaPhi->Write();
  if (numEvents > 10) {
    // Skip first 10 entries
    hTimers->Scale(1. / (numEvents - 10.));
  }
  hTimers->Write();
  delete MyFile;
}

void PFClusterProducerCudaECAL::beginRun(const edm::Run &run, const edm::EventSetup &es) {
  _initialClustering->update(es);
  if (_pfClusterBuilder)
    _pfClusterBuilder->update(es);
  if (_positionReCalc)
    _positionReCalc->update(es);
}

void PFClusterProducerCudaECAL::acquire(edm::Event const &event,
                                        edm::EventSetup const &setup,
                                        edm::WaitingTaskWithArenaHolder holder) {
  // Creates a new Cuda stream
  // TODO: Reuse stream from GPU PFRecHitProducer by passing input product as first arg
  // cmssdt.cern.ch/lxr/source/HeterogeneousCore/CUDACore/interface/ScopedContext.h#0101
  //cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder)};

  cudaStream = ctx.stream();

  if (numEvents == 0) {
    // Initialize Cuda stuff here
    PFClusterCudaECAL::initializeCudaConstants(cudaConstants, cudaStream);

    inputCPU.allocate(cudaConfig_, cudaStream);
    outputCPU.allocate(cudaConfig_, cudaStream);
    inputGPU.allocate(cudaConfig_, cudaStream);
    outputGPU.allocate(cudaConfig_, cudaStream);
    scratchGPU.allocate(cudaConfig_, cudaStream);
  }

  _initialClustering->reset();
  if (_pfClusterBuilder)
    _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  event.getByToken(_rechitsLabel, rechits);

#ifdef DEBUG_ECAL_TREES
  GPU_timers.fill(0.0);
  __pfcIter.clear();
  __nRHTopo.clear();
  __nSeedsTopo.clear();
  __nFracsTopo.clear();
  __rechits = *rechits;
  __rh_mask.clear();
  __rh_isSeed.clear();
  __rh_x.clear();
  __rh_y.clear();
  __rh_z.clear();
  __rh_eta.clear();
  __rh_phi.clear();
  __rh_pt2.clear();
  __rh_neighbours8.clear();
  __rh_axis_x.clear();
  __rh_axis_y.clear();
  __rh_axis_z.clear();
#endif

  _initialClustering->updateEvent(event);

  std::vector<bool> mask(rechits->size(), true);
  for (const auto &cleaner : _cleaners) {
    cleaner->clean(rechits, mask);
  }

#ifdef DEBUG_ECAL_TREES
  for (auto isMasked : mask) {
    __rh_mask.push_back(isMasked);
  }
#endif

  int p = 0;
  int totalNeighbours = 0;  // Running count of 8 neighbour edges for edgeId, edgeList
  int nRH = (int)rechits->size();
  for (const auto &rh : *rechits) {
    //std::cout<<"*** Now on rechit \t"<<p<<"\tdetId = "<<rh.detId()<<"\tneighbourInfos().size() = "<<rh.neighbourInfos().size()<<"\tneighbours8().size() = "<<rh.neighbours8().size()<<std::endl;

    // https://cmssdt.cern.ch/lxr/source/Geometry/CaloGeometry/src/TruncatedPyramid.cc#0057
    auto corners = rh.getCornersXYZ();
    auto backCtr = GlobalPoint(0.25 * (corners[4].x() + corners[5].x() + corners[6].x() + corners[7].x()),
                               0.25 * (corners[4].y() + corners[5].y() + corners[6].y() + corners[7].y()),
                               0.25 * (corners[4].z() + corners[5].z() + corners[6].z() + corners[7].z()));
    auto axis = GlobalVector(backCtr - GlobalPoint(rh.position())).unit();

    inputCPU.rh_axis_x[p] = axis.x();
    inputCPU.rh_axis_y[p] = axis.y();
    inputCPU.rh_axis_z[p] = axis.z();

    inputCPU.pfrh_x[p] = rh.position().x();
    inputCPU.pfrh_y[p] = rh.position().y();
    inputCPU.pfrh_z[p] = rh.position().z();
    inputCPU.pfrh_energy[p] = rh.energy();
    inputCPU.pfrh_pt2[p] = rh.pt2();
    inputCPU.pfrh_layer[p] = (int)rh.layer();

#ifdef DEBUG_ECAL_TREES
    __rh_axis_x.push_back(axis.x());
    __rh_axis_y.push_back(axis.y());
    __rh_axis_z.push_back(axis.z());
    __rh_x.push_back(inputCPU.pfrh_x[p]);
    __rh_y.push_back(inputCPU.pfrh_y[p]);
    __rh_z.push_back(inputCPU.pfrh_z[p]);
    __rh_eta.push_back(rh.positionREP().eta());
    __rh_phi.push_back(rh.positionREP().phi());
    __rh_pt2.push_back(inputCPU.pfrh_pt2[p]);
#endif

    std::vector<unsigned int> n8;
    auto theneighboursEight = rh.neighbours8();
    int z = 0;
    for (auto nh : theneighboursEight) {
      n8.push_back(nh);
    }
    std::sort(n8.begin(), n8.end());  // Sort 8 neighbour edges in ascending order for topo clustering
    for (auto nh : n8) {
      inputCPU.pfNeighEightInd[8 * p + z] = nh;
      inputCPU.pfrh_edgeId[totalNeighbours] = p;
      inputCPU.pfrh_edgeList[totalNeighbours] = (int)nh;
      totalNeighbours++;
      z++;
    }
    for (int l = z; l < 8; l++) {
      inputCPU.pfNeighEightInd[8 * p + l] = -1;
    }

    p++;
#ifdef DEBUG_ECAL_TREES
    __rh_neighbours8.push_back(n8);
#endif
  }  //end of rechit loop

  nEdges = totalNeighbours;

#ifdef DEBUG_GPU_ECAL
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, cudaStream);
#endif

  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_x.get(), inputCPU.pfrh_x.get(), sizeof(float) * nRH, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_y.get(), inputCPU.pfrh_y.get(), sizeof(float) * nRH, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_z.get(), inputCPU.pfrh_z.get(), sizeof(float) * nRH, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_energy.get(), inputCPU.pfrh_energy.get(), sizeof(float) * nRH, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_pt2.get(), inputCPU.pfrh_pt2.get(), sizeof(float) * nRH, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_layer.get(), inputCPU.pfrh_layer.get(), sizeof(int) * nRH, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfNeighEightInd.get(),
                            inputCPU.pfNeighEightInd.get(),
                            sizeof(int) * nRH * 8,
                            cudaMemcpyHostToDevice,
                            cudaStream));

  cudaCheck(cudaMemcpyAsync(
      inputGPU.rh_axis_x.get(), inputCPU.rh_axis_x.get(), sizeof(float) * nRH, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.rh_axis_y.get(), inputCPU.rh_axis_y.get(), sizeof(float) * nRH, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.rh_axis_z.get(), inputCPU.rh_axis_z.get(), sizeof(float) * nRH, cudaMemcpyHostToDevice, cudaStream));

  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_edgeId.get(),
                            inputCPU.pfrh_edgeId.get(),
                            sizeof(int) * totalNeighbours,
                            cudaMemcpyHostToDevice,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_edgeList.get(),
                            inputCPU.pfrh_edgeList.get(),
                            sizeof(int) * totalNeighbours,
                            cudaMemcpyHostToDevice,
                            cudaStream));

#ifdef DEBUG_GPU_ECAL
  cudaEventRecord(stop, cudaStream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[0], start, stop);
  //std::cout<<"(ECAL) Copy memory to device: "<<GPU_timers[0]<<" ms"<<std::endl;
#endif

  float kernelTimers[8] = {0.0};

  /*  
  //PFClusterCudaECAL::PFRechitToPFCluster_ECAL_serialize(rechits->size(), 
  PFClusterCudaECAL::PFRechitToPFCluster_ECALV2(rechits->size(), 
                          inputGPU.pfrh_x.get(),
                          inputGPU.pfrh_y.get(),
                          inputGPU.pfrh_z.get(),
                          inputGPU.pfrh_energy.get(),
                          inputGPU.pfrh_pt2.get(),
                          inputGPU.pfrh_isSeed.get(),
                          inputGPU.pfrh_topoId.get(),
                          inputGPU.pfrh_layer.get(),
                          inputGPU.pfNeighEightInd.get(),
                          inputGPU.pcrh_fracInd.get(),
                          inputGPU.pcrh_frac.get(),
                          inputGPU.pcrh_fracSum.get(),
                          inputGPU.rhcount.get(),
                          kernelTimers
					      );
  */

  PFClusterCudaECAL::PFRechitToPFCluster_ECAL_CCLClustering(cudaStream,
                                                            nRH,
                                                            (int)totalNeighbours,
                                                            inputGPU.pfrh_x.get(),
                                                            inputGPU.pfrh_y.get(),
                                                            inputGPU.pfrh_z.get(),
                                                            inputGPU.rh_axis_x.get(),
                                                            inputGPU.rh_axis_y.get(),
                                                            inputGPU.rh_axis_z.get(),
                                                            inputGPU.pfrh_energy.get(),
                                                            inputGPU.pfrh_pt2.get(),
                                                            outputGPU.pfrh_isSeed.get(),
                                                            outputGPU.pfrh_topoId.get(),
                                                            inputGPU.pfrh_layer.get(),
                                                            inputGPU.pfNeighEightInd.get(),
                                                            inputGPU.pfrh_edgeId.get(),
                                                            inputGPU.pfrh_edgeList.get(),
                                                            inputGPU.pfrh_edgeMask.get(),
                                                            inputGPU.pfrh_passTopoThresh.get(),
                                                            outputGPU.pcrh_fracInd.get(),
                                                            outputGPU.pcrh_frac.get(),
                                                            scratchGPU.pcrh_fracSum.get(),
                                                            scratchGPU.rhcount.get(),
                                                            outputGPU.topoSeedCount.get(),
                                                            outputGPU.topoRHCount.get(),
                                                            outputGPU.seedFracOffsets.get(),
                                                            outputGPU.topoSeedOffsets.get(),
                                                            outputGPU.topoSeedList.get(),
                                                            scratchGPU.pfc_pos4.get(),
                                                            scratchGPU.pfc_prevPos4.get(),
                                                            scratchGPU.pfc_linearPos4.get(),
                                                            scratchGPU.pfc_convPos4.get(),
                                                            scratchGPU.pfc_energy.get(),
                                                            scratchGPU.pfc_clusterT0.get(),
                                                            kernelTimers,
                                                            outputGPU.topoIter.get(),
                                                            outputGPU.pfc_iter.get(),
                                                            outputGPU.pcrhFracSize.get());

  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoIter.get(), outputGPU.topoIter.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pcrhFracSize.get(), outputGPU.pcrhFracSize.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));
  // Total size of allocated rechit fraction arrays (includes some extra padding for rechits that don't end up passing cuts)
  nFracs = outputCPU.pcrhFracSize[0];
  //  std::cout<<"ECAL: nFracs = "<<nFracs<<std::endl;

#ifdef DEBUG_GPU_ECAL
  GPU_timers[1] = kernelTimers[0];
  GPU_timers[2] = kernelTimers[1];
  GPU_timers[3] = kernelTimers[2];
  GPU_timers[4] = kernelTimers[3];
  //  std::cout<<"ECAL GPU clustering (ms):\n"
  //           <<"Seeding\t\t"<<GPU_timers[1]<<std::endl
  //           <<"Topo clustering\t"<<GPU_timers[2]<<std::endl
  //           <<"PF cluster step 1 \t"<<GPU_timers[3]<<std::endl
  //           <<"PF cluster step 2 \t"<<GPU_timers[4]<<std::endl;
  cudaEventRecord(start, cudaStream);
#endif
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfc_iter.get(), outputGPU.pfc_iter.get(), sizeof(int) * nRH, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(outputCPU.topoSeedCount.get(),
                            outputGPU.topoSeedCount.get(),
                            sizeof(int) * nRH,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoRHCount.get(), outputGPU.topoRHCount.get(), sizeof(int) * nRH, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(outputCPU.seedFracOffsets.get(),
                            outputGPU.seedFracOffsets.get(),
                            sizeof(int) * nRH,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(outputCPU.topoSeedOffsets.get(),
                            outputGPU.topoSeedOffsets.get(),
                            sizeof(int) * nRH,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(outputCPU.topoSeedList.get(),
                            outputGPU.topoSeedList.get(),
                            sizeof(int) * nRH,
                            cudaMemcpyDeviceToHost,
                            cudaStream));

  cudaCheck(cudaMemcpyAsync(outputCPU.pcrh_fracInd.get(),
                            outputGPU.pcrh_fracInd.get(),
                            sizeof(int) * nFracs,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pcrh_frac.get(), outputGPU.pcrh_frac.get(), sizeof(float) * nFracs, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfrh_isSeed.get(), outputGPU.pfrh_isSeed.get(), sizeof(int) * nRH, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfrh_topoId.get(), outputGPU.pfrh_topoId.get(), sizeof(int) * nRH, cudaMemcpyDeviceToHost, cudaStream));

  //  bool*                                                 h_cuda_pfrh_passTopoThresh = new bool[rechits->size()];
  //  cudaCheck(cudaMemcpyAsync(h_cuda_pfrh_passTopoThresh, inputGPU.pfrh_passTopoThresh.get(), sizeof(bool)*rechits->size(), cudaMemcpyDeviceToHost));

#ifdef DEBUG_GPU_ECAL
  cudaEventRecord(stop, cudaStream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[5], start, stop);
//  std::cout<<"(ECAL) Copy results from GPU: "<<GPU_timers[5]<<" ms"<<std::endl;
#endif
  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

#ifdef DEBUG_GPU_ECAL
  nFracs_vs_nRH->Fill(nRH, nFracs);
  for (int topoId = 0; topoId < nRH; topoId++) {
    int nPFCIter = outputCPU.pfc_iter[topoId];  // Number of iterations for PF clustering to converge
    int nSeeds = outputCPU.topoSeedCount[topoId];
    int nRHTopo = outputCPU.topoRHCount[topoId];
    int nFracsTopo = (nRHTopo - nSeeds + 1) * nSeeds;
#ifdef DEBUG_ECAL_TREES
    __pfcIter.push_back(nPFCIter);
    __nRHTopo.push_back(nRHTopo);
    __nSeedsTopo.push_back(nSeeds);
    __nFracsTopo.push_back(nFracsTopo);
#endif
    if (nPFCIter >= 0) {
      pfcIterations->Fill(nPFCIter);
      pfcIter_vs_nRHTopo->Fill(nRHTopo, nPFCIter);
      pfcIter_vs_nSeedsTopo->Fill(nSeeds, nPFCIter);
      pfcIter_vs_nFracsTopo->Fill(nFracsTopo, nPFCIter);  // Total number of rechit fractions for a given topo cluster
    }
  }
#endif

  topoIter = outputCPU.topoIter[0];
  topoIterations->Fill(topoIter);
  topoIter_vs_nRH->Fill(rechits->size(), topoIter);

  pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();

  for (int n = 0; n < nRH; n++) {
#ifdef DEBUG_HCAL_TREES
    __rh_isSeed.push_back(outputCPU.pfrh_isSeed[n]);
#endif
    if (outputCPU.pfrh_isSeed[n] == 1 && outputCPU.pfrh_topoId[n] > -1) {
      reco::PFCluster temp;
      temp.setSeed((*rechits)[n].detId());
      int offset = outputCPU.seedFracOffsets[n];
      int topoId = outputCPU.pfrh_topoId[n];
      int nSeeds = outputCPU.topoSeedCount[topoId];
      //std::cout<<"\nCluster (seed "<<n<<")\toffset = "<<offset<<"\ttopoId = "<<topoId<<"\tnSeeds = "<<nSeeds<<"\tnFrac = "<<outputCPU.topoRHCount[topoId] - nSeeds + 1<<"\tlayer = "<<inputCPU.pfrh_layer[n]<<std::endl;
      for (int k = offset; k < (offset + outputCPU.topoRHCount[topoId] - nSeeds + 1); k++) {
        //std::cout<<"\tNow on k = "<<k<<"\tindex = "<<outputCPU.pcrh_fracInd[k]<<"\tfrac = "<<outputCPU.pcrh_frac[k]<<std::endl;
        if (outputCPU.pcrh_fracInd[k] > -1 && outputCPU.pcrh_frac[k] > 0.0) {
          const reco::PFRecHitRef &refhit = reco::PFRecHitRef(rechits, outputCPU.pcrh_fracInd[k]);
          temp.addRecHitFraction(reco::PFRecHitFraction(refhit, outputCPU.pcrh_frac[k]));
        }
      }
      pfClustersFromCuda->push_back(temp);
    }
  }
  _positionReCalc->calculateAndSetPositions(*pfClustersFromCuda);

  if (doComparison) {
    std::vector<bool> seedable(rechits->size(), false);
    _seedFinder->findSeeds(rechits, mask, seedable);
#ifdef DEBUG_ECAL_TREES
    for (auto isSeed : seedable) {
      __rh_isSeed.push_back((int)isSeed);
    }
#endif
    auto initialClusters = std::make_unique<reco::PFClusterCollection>();
    _initialClustering->buildClusters(rechits, mask, seedable, *initialClusters);
    LOGVERB("PFClusterProducer::produce()") << *_initialClustering;
#if defined DEBUG_ECAL_TREES && defined DEBUG_SAVE_CLUSTERS
    __initialClusters = *initialClusters;  // For TTree
#endif
    int topoRhCount = 0;
    int clusterCount = 0;
    for (const auto &pfc : *initialClusters) {
      nTopo_CPU->Fill(pfc.recHitFractions().size());
      /*
        std::cout<<"Cluster "<<clusterCount<<" has "<<pfc.recHitFractions().size()<<" rechits"<<std::endl;
        for (auto rhf : pfc.recHitFractions()) {
            std::cout<<"rhf.recHitRef().index() = "<<rhf.recHitRef().index()<<"\trhf.recHitRef()->detId() = "<<rhf.recHitRef()->detId()<<"\trhf.recHitRef().get() = "<<rhf.recHitRef().get()<<std::endl;
        }
        std::cout<<std::endl;
        */
      topoEn_CPU->Fill(pfc.energy());
      topoEta_CPU->Fill(pfc.eta());
      topoPhi_CPU->Fill(pfc.phi());

      topoRhCount = topoRhCount + pfc.recHitFractions().size();
      int nSeeds = 0;
      for (const auto &rhf : pfc.recHitFractions()) {
        if (seedable[rhf.recHitRef().key()])
          nSeeds++;
      }
      topoSeeds_CPU->Fill(nSeeds);

      //std::cout<<"Cluster "<<clusterCount<<" has "<<pfc.recHitFractions().size()<<" rechit fractions"<<std::endl;
      //        for (auto rhf : pfc.recHitFractions())
      //        {
      //            auto rh = *rhf.recHitRef().get();
      //            //std::cout<<"detId = "<<rh.detId()<<"\teta = "<<rh.position().eta()<<"\tphi = "<<rh.position().phi()<<std::endl;
      //
      //        }
      //std::cout<<std::endl<<std::endl;
      clusterCount++;
    }

    nPFCluster_CPU->Fill(initialClusters->size());

    std::unordered_map<int, std::vector<int>> nTopoRechits;
    std::unordered_map<int, int> nTopoSeeds;

    for (int rh = 0; rh < (int)rechits->size(); rh++) {
      int topoId = outputCPU.pfrh_topoId[rh];
      if (topoId > -1) {
        // Valid topo id
        nTopoRechits[topoId].push_back(rh);
        if (outputCPU.pfrh_isSeed[rh] > 0) {
          nTopoSeeds[topoId]++;
        }
      }
    }
    /* 
    for(unsigned int i=0;i<nRH;i++){
      int topoIda=h_cuda_pfrh_topoId[i];
      if (nTopoSeeds.count(topoIda) == 0) continue;
      for(unsigned int j=0;j<8;j++){
        if(h_cuda_pfNeighEightInd[i*8+j]>-1 && h_cuda_pfrh_topoId[h_cuda_pfNeighEightInd[i*8+j]]!=topoIda && h_cuda_pfrh_passTopoThresh[i*8+j]) std::cout<<"ECAL HAS DIFFERENT TOPOID "<<i<<"  "<<j<<"  "<<topoIda<<"  "<<h_cuda_pfrh_topoId[h_cuda_pfNeighEightInd[i*8+j]]<<std::endl;
      }
    }
  */

    int intTopoCount = 0;
    for (const auto &x : nTopoRechits) {
      int topoId = x.first;
      if (nTopoSeeds.count(topoId) > 0) {
        // This topo cluster has at least one seed
        nTopo_GPU->Fill(x.second.size());
        topoSeeds_GPU->Fill(nTopoSeeds[topoId]);
        intTopoCount++;
      }
    }

    nPFCluster_GPU->Fill(intTopoCount);
    //std::sort (h_cuda_pfrh_topoId.begin(), h_cuda_pfrh_topoId.end());

    int seedSumCPU = 0;
    int seedSumGPU = 0;
    int maskSize = 0;
    for (int j = 0; j < (int)seedable.size(); j++)
      seedSumCPU = seedSumCPU + seedable[j];
    for (int j = 0; j < nRH; j++)
      seedSumGPU = seedSumGPU + outputCPU.pfrh_isSeed[j];
    for (int j = 0; j < (int)mask.size(); j++)
      maskSize = maskSize + mask[j];

    //std::cout<<"ECAL sum CPU seeds: "<<seedSumCPU<<std::endl;

    sumSeed_CPU->Fill(seedSumCPU);
    sumSeed_GPU->Fill(seedSumGPU);
    deltaSumSeed->Fill(seedSumGPU - seedSumCPU);

    auto pfClusters = std::make_unique<reco::PFClusterCollection>();
    pfClusters = std::make_unique<reco::PFClusterCollection>();
    if (_pfClusterBuilder) {  // if we've defined a re-clustering step execute it
      _pfClusterBuilder->buildClusters(*initialClusters, seedable, *pfClusters);
      LOGVERB("PFClusterProducer::produce()") << *_pfClusterBuilder;
    } else {
      pfClusters->insert(pfClusters->end(), initialClusters->begin(), initialClusters->end());
    }

    int totalRHPF_CPU = 0, totalRHPF_GPU = 0;
#if defined DEBUG_ECAL_TREES && defined DEBUG_SAVE_CLUSTERS
    __pfClusters = *pfClusters;  // For TTree
#endif
    for (const auto &pfc : *pfClusters) {
      nRH_perPFCluster_CPU->Fill(pfc.recHitFractions().size());
      totalRHPF_CPU += (int)pfc.recHitFractions().size();
      enPFCluster_CPU->Fill(pfc.energy());
      pfcEta_CPU->Fill(pfc.eta());
      pfcPhi_CPU->Fill(pfc.phi());
      for (const auto &pfcx : *pfClustersFromCuda) {
        if (pfc.seed() == pfcx.seed()) {
          totalRHPF_GPU += (int)pfcx.recHitFractions().size();

          matched_pfcRh_CPU->Fill(pfc.recHitFractions().size());
          matched_pfcRh_GPU->Fill(pfcx.recHitFractions().size());
          matched_pfcEn_CPU->Fill(pfc.energy());
          matched_pfcEn_GPU->Fill(pfcx.energy());
          matched_pfcEta_CPU->Fill(pfc.eta());
          matched_pfcEta_GPU->Fill(pfcx.eta());
          matched_pfcPhi_CPU->Fill(pfc.phi());
          matched_pfcPhi_GPU->Fill(pfcx.phi());

          if (abs((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size()) > 0) {
            std::cout << "ECAL mismatch nRH:\tGPU:" << (int)pfcx.recHitFractions().size()
                      << "\tCPU:" << (int)pfc.recHitFractions().size() << std::endl;
          }
          deltaRH->Fill((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size());
          if (abs(pfcx.energy() - pfc.energy()) > 1e-2) {
            std::cout << "ECAL mismatch  En:\tGPU:" << pfcx.energy() << "\tCPU:" << pfc.energy() << std::endl;
          }
          deltaEn->Fill(pfcx.energy() - pfc.energy());
          if (abs(pfcx.eta() - pfc.eta()) > 1e-4) {
            std::cout << "ECAL mismatch Eta:\tGPU:" << pfcx.eta() << "\tCPU:" << pfc.eta() << std::endl;
          }
          deltaEta->Fill(pfcx.eta() - pfc.eta());
          if (abs(pfcx.phi() - pfc.phi()) > 1e-4) {
            std::cout << "ECAL mismatch Phi:\tGPU:" << pfcx.phi() << "\tCPU:" << pfc.phi() << std::endl;
          }
          deltaPhi->Fill(pfcx.phi() - pfc.phi());

          nRh_CPUvsGPU->Fill(pfcx.recHitFractions().size(), pfc.recHitFractions().size());
          enPFCluster_CPUvsGPU->Fill(pfcx.energy(), pfc.energy());

          if (abs((pfcx.energy() - pfc.energy()) / pfc.energy()) > 0.05) {
            coordinate->Fill(pfcx.eta(), pfcx.phi());
            for (const auto &rhf : pfc.recHitFractions()) {
              if (rhf.fraction() == 1)
                layer->Fill(rhf.recHitRef()->depth());
            }
          }
        }
      }
    }

    nRH_perPFClusterTotal_CPU->Fill(totalRHPF_CPU);
    nRH_perPFClusterTotal_GPU->Fill(totalRHPF_GPU);

    nRHperPFCTotal_CPU = totalRHPF_CPU;
    nRHperPFCTotal_GPU = totalRHPF_GPU;

#if defined DEBUG_ECAL_TREES && defined DEBUG_SAVE_CLUSTERS
    __pfClustersFromCuda = *pfClustersFromCuda;  // For TTree
#endif
    for (const auto &pfc : *pfClustersFromCuda) {
      nRH_perPFCluster_GPU->Fill(pfc.recHitFractions().size());
      enPFCluster_GPU->Fill(pfc.energy());
      pfcEta_GPU->Fill(pfc.eta());
      pfcPhi_GPU->Fill(pfc.phi());
    }
  }

#ifdef DEBUG_GPU_ECAL
  if (numEvents > 9) {
    for (int i = 0; i < (int)GPU_timers.size(); i++)
      hTimers->Fill(i, GPU_timers[i]);
  }
#endif

#ifdef DEBUG_ECAL_TREES
  clusterTree->Fill();
#endif
  numEvents++;
}

void PFClusterProducerCudaECAL::produce(edm::Event &event, const edm::EventSetup &setup) {
  //std::cout<<"\n===== Now on event "<<numEvents<<" with "<<rechits->size()<<" ECAL rechits ====="<<std::endl;
  if (_prodInitClusters)
    event.put(std::move(pfClustersFromCuda), "initialClusters");
  event.put(std::move(pfClustersFromCuda));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFClusterProducerCudaECAL);
