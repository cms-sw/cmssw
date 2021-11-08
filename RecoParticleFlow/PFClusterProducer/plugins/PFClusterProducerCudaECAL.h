#ifndef __newpf_PFClusterProducerCudaECAL_H__
#define __newpf_PFClusterProducerCudaECAL_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"


#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/CudaPFCommon.h"

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"

#include <memory>
#include <array>
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
  
      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = 0) {
        pfrh_x = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_y = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_z = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_energy = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_pt2 = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        
        // Detector geometry
        rh_axis_x = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        rh_axis_y = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        rh_axis_z = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        
        pfrh_layer = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfNeighEightInd = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_edgeId = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_edgeList = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
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

      cms::cuda::host::unique_ptr<int[]> pfc_iter; // Iterations per pf cluster (by seed index)
      cms::cuda::host::unique_ptr<int[]> topoIter;  // Iterations for topo clustering to converge
      cms::cuda::host::unique_ptr<int[]> pcrhFracSize;  // Total number of pfc fractions to copy back
      
      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = 0) {
        pfrh_topoId = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfrh_isSeed = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pcrh_frac = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxPFCFracs, cudaStream);
        pcrh_fracInd = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxPFCFracs, cudaStream);
        pfrh_passTopoThresh = cms::cuda::make_host_unique<bool[]>(sizeof(bool)*config.maxRH, cudaStream);
        
        topoSeedCount = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        topoRHCount = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        seedFracOffsets = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        topoSeedOffsets = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        topoSeedList = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        
        pfc_iter = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
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
      
      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = 0) {
        pfrh_x = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_y = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_z = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_energy = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_pt2 = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        
        // Detector geometry
        rh_axis_x = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        rh_axis_y = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        rh_axis_z = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
       

        pfrh_layer = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfNeighEightInd = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_edgeId = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_edgeList = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_edgeMask = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_passTopoThresh = cms::cuda::make_device_unique<bool[]>(sizeof(bool)*config.maxRH, cudaStream);
      }
  };
  
  struct OutputDataGPU {
      cms::cuda::device::unique_ptr<int[]> pfrh_topoId;
      cms::cuda::device::unique_ptr<int[]> pfrh_isSeed;
      cms::cuda::device::unique_ptr<float[]> pcrh_frac;
      cms::cuda::device::unique_ptr<int[]>   pcrh_fracInd;
      cms::cuda::device::unique_ptr<bool[]> pfrh_passTopoThresh;
      
      cms::cuda::device::unique_ptr<int[]> topoSeedCount;
      cms::cuda::device::unique_ptr<int[]> topoRHCount;
      cms::cuda::device::unique_ptr<int[]> seedFracOffsets;
      cms::cuda::device::unique_ptr<int[]> topoSeedOffsets;
      cms::cuda::device::unique_ptr<int[]> topoSeedList;

      cms::cuda::device::unique_ptr<int[]> pfc_iter; // Iterations per pf cluster (by seed index)
      cms::cuda::device::unique_ptr<int[]> topoIter;  // Iterations for topo clustering to converge
      cms::cuda::device::unique_ptr<int[]> pcrhFracSize;  // Total number of pfc fractions to copy back
      
      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = 0) {
        pfrh_topoId = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfrh_isSeed = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pcrh_frac = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxPFCFracs, cudaStream);
        pcrh_fracInd = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxPFCFracs, cudaStream);
        pfrh_passTopoThresh = cms::cuda::make_device_unique<bool[]>(sizeof(bool)*config.maxRH, cudaStream);
        
        topoSeedCount = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        topoRHCount = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        seedFracOffsets = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        topoSeedOffsets = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        topoSeedList = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        
        pfc_iter = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        topoIter = cms::cuda::make_device_unique<int[]>(sizeof(int), cudaStream);
        pcrhFracSize = cms::cuda::make_device_unique<int[]>(sizeof(int), cudaStream);
      }
  };

  struct ScratchDataGPU {
      cms::cuda::device::unique_ptr<float4[]> pfc_convPos4;   // Convergence position 
      cms::cuda::device::unique_ptr<float4[]> pfc_prevPos4;   // Previous (convergence) position 
      cms::cuda::device::unique_ptr<float4[]> pfc_linearPos4; // Linear position
      cms::cuda::device::unique_ptr<float4[]> pfc_pos4;       // Cluster position
      cms::cuda::device::unique_ptr<float[]> pfc_energy;      // Cluster energy
      cms::cuda::device::unique_ptr<float[]> pfc_clusterT0;   // Cluster T0 (for 2D position calculation)
      
      cms::cuda::device::unique_ptr<int[]> rhcount;
      cms::cuda::device::unique_ptr<float[]> pcrh_fracSum;
      
      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = 0 /* default Cuda stream */) {
        pfc_convPos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4)*config.maxRH, cudaStream);
        pfc_prevPos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4)*config.maxRH, cudaStream);
        pfc_linearPos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4)*config.maxRH, cudaStream);
        pfc_pos4 = cms::cuda::make_device_unique<float4[]>(sizeof(float4)*config.maxRH, cudaStream);
        pfc_energy = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfc_clusterT0 = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        
        rhcount = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pcrh_fracSum = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
      }
  };
 
 } // namespace ECAL
} // namespace PFClustering

class PFClusterProducerCudaECAL : public edm::stream::EDProducer<edm::ExternalWork> {
  typedef RecHitTopologicalCleanerBase RHCB;
  typedef InitialClusteringStepBase ICSB;
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;

public:
  PFClusterProducerCudaECAL(const edm::ParameterSet&);
  ~PFClusterProducerCudaECAL();

  //void endJob();
  //  void beginStream(edm::StreamID);
 
  void initializeCudaMemory(cudaStream_t cudaStream);
  void freeCudaMemory(); 
  
 

  // options
  const bool _prodInitClusters;
  // the actual algorithm
  std::vector<std::unique_ptr<RecHitTopologicalCleanerBase> > _cleaners;
  std::unique_ptr<SeedFinderBase> _seedFinder;
  std::unique_ptr<InitialClusteringStepBase> _initialClustering;
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFCPositionCalculatorBase> _positionReCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _convergencePosCalc;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;

  TFile *MyFile = new TFile("EventECAL.root","recreate");
  
  reco::PFClusterCollection __initialClusters;
  reco::PFClusterCollection __pfClusters;
  reco::PFClusterCollection __pfClustersFromCuda;
  reco::PFRecHitCollection  __rechits;

  // rechit physics quantities 
  std::vector<int>    __rh_mask;
  std::vector<int>    __rh_isSeed;
  std::vector<float>  __rh_x;
  std::vector<float>  __rh_y;
  std::vector<float>  __rh_z;
  std::vector<float>  __rh_axis_x;
  std::vector<float>  __rh_axis_y;
  std::vector<float>  __rh_axis_z;
  std::vector<float>  __rh_eta;
  std::vector<float>  __rh_phi;
  std::vector<float> __rh_pt2;
  // rechit neighbours4, neighbours8 vectors
  std::vector<std::vector<unsigned int>> __rh_neighbours8;

  std::vector<int>    __pfcIter;
  std::vector<int>    __nRHTopo;
  std::vector<int>    __nSeedsTopo;
  std::vector<int>    __nFracsTopo;

  TTree *clusterTree = new TTree("clusterTree", "clusterTree");
  
  TH1F *pfcIterations = new TH1F("pfcIter","nIterations PF Clustering", 50, 0.5, 50.5);
  TH2F *pfcIter_vs_nRHTopo = new TH2F("pfcIternRHTopo","nIterations PF Clustering vs nRH topo", 300, 0.5, 300.5, 50, 0.5, 50.5);
  TH2F *pfcIter_vs_nSeedsTopo = new TH2F("pfcIternSeedsTopo","nIterations PF Clustering vs nSeeds in topo", 75, 0.5, 75.5, 50, 0.5, 50.5);
  TH2F *pfcIter_vs_nFracsTopo = new TH2F("pfcIternFracsTopo","nIterations PF Clustering vs nFracs in topo", 260, 0.5, 260.5, 50, 0.5, 50.5);

  TH2F *nFracs_vs_nRH = new TH2F("nFracsnRH","PF Cluster Fractions vs num rechits", 50, 0.5, 2000.5, 50, 0.5, 70000.5);
  
  TH1F *topoIterations = new TH1F("topoIter","nIterations Topo Clustering", 25, 0.5, 25.5);
  TH2F *topoIter_vs_nRH = new TH2F("topoIternRH","nIterations Topo Clustering vs num rechits", 3000, 0.5, 3000.5, 25, 0.5, 25.5);
  TH1F *nTopo_CPU = new TH1F("nTopo_CPU","nTopo_CPU",500,0.5,500.5);
  TH1F *nTopo_GPU = new TH1F("nTopo_GPU","nTopo_GPU",500,0.5,500.5);

  TH1F *topoSeeds_CPU = new TH1F("topoSeeds_CPU","topoSeeds_CPU",200,0.5,200.5);
  TH1F *topoSeeds_GPU = new TH1F("topoSeeds_GPU","topoSeeds_GPU",200,0.5,200.5);

  TH1F *sumSeed_CPU = new TH1F("sumSeed_CPU", "sumSeed_CPU",200, 0.5, 200.5);
  TH1F *sumSeed_GPU = new TH1F("sumSeed_GPU", "sumSeed_GPU",200, 0.5, 200.5);

  TH1F *topoEn_CPU = new TH1F("topoEn_CPU", "topoEn_CPU", 500, 0, 500);
  TH1F *topoEn_GPU = new TH1F("topoEn_GPU", "topoEn_GPU", 500, 0, 500);
  
  TH1F *topoEta_CPU = new TH1F("topoEta_CPU", "topoEta_CPU", 100, -3, 3);
  TH1F *topoEta_GPU = new TH1F("topoEta_GPU", "topoEta_GPU", 100, -3, 3);
  
  TH1F *topoPhi_CPU = new TH1F("topoPhi_CPU", "topoPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *topoPhi_GPU = new TH1F("topoPhi_GPU", "topoPhi_GPU", 100, -3.1415926, 3.1415926);

  TH1F *nPFCluster_CPU = new TH1F("nPFCluster_CPU","nPFCluster_CPU",1000,0.5,1000.5);
  TH1F *nPFCluster_GPU = new TH1F("nPFCluster_GPU","nPFCluster_GPU",1000,0.5,1000.5);

  TH1F *enPFCluster_CPU = new TH1F("enPFCluster_CPU","enPFCluster_CPU",500,0,500);
  TH1F *enPFCluster_GPU = new TH1F("enPFCluster_GPU","enPFCluster_GPU",500,0,500);

  TH1F *pfcEta_CPU = new TH1F("pfcEta_CPU", "pfcEta_CPU", 100, -3, 3);
  TH1F *pfcEta_GPU = new TH1F("pfcEta_GPU", "pfcEta_GPU", 100, -3, 3);

  TH1F *pfcPhi_CPU = new TH1F("pfcPhi_CPU", "pfcPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *pfcPhi_GPU = new TH1F("pfcPhi_GPU", "pfcPhi_GPU", 100, -3.1415926, 3.1415926);

  TH1F *nRH_perPFCluster_CPU = new TH1F("nRH_perPFCluster_CPU","nRH_perPFCluster_CPU",101,-0.5,100.5);
  TH1F *nRH_perPFCluster_GPU = new TH1F("nRH_perPFCluster_GPU","nRH_perPFCluster_GPU",101,-0.5,100.5);
  
  // Total number of rechit fractions in all PF clusters per event (includes float counting)
  TH1F *nRH_perPFClusterTotal_CPU = new TH1F("nRH_perPFClusterTotal_CPU","nRH_perPFClusterTotal_CPU",2000,0.5,2000.5);
  TH1F *nRH_perPFClusterTotal_GPU = new TH1F("nRH_perPFClusterTotal_GPU","nRH_perPFClusterTotal_GPU",2000,0.5,2000.5);

  TH1F *matched_pfcRh_CPU = new TH1F("matched_pfcRh_CPU", "matching seed pfcRh_CPU", 101,-0.5,100.5);
  TH1F *matched_pfcRh_GPU = new TH1F("matched_pfcRh_GPU", "matching seed pfcRh_GPU", 101,-0.5,100.5);

  TH1F *matched_pfcEn_CPU = new TH1F("matched_pfcEn_CPU", "matching seed pfcEn_CPU", 500,0,500);  
  TH1F *matched_pfcEn_GPU = new TH1F("matched_pfcEn_GPU", "matching seed pfcEn_GPU", 500,0,500);

  TH1F *matched_pfcEta_CPU = new TH1F("matched_pfcEta_CPU", "matching seed pfcEta_CPU", 100, -3, 3);
  TH1F *matched_pfcEta_GPU = new TH1F("matched_pfcEta_GPU", "matching seed pfcEta_GPU", 100, -3, 3);

  TH1F *matched_pfcPhi_CPU = new TH1F("matched_pfcPhi_CPU", "matching seed pfcPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *matched_pfcPhi_GPU = new TH1F("matched_pfcPhi_GPU", "matching seed pfcPhi_GPU", 100, -3.1415926, 3.1415926);

  TH2F *nRh_CPUvsGPU = new TH2F("nRh_CPUvsGPU","nRh_CPUvsGPU",101,-0.5,100.5,101,-0.5,100.5);
  TH2F *enPFCluster_CPUvsGPU = new TH2F("enPFCluster_CPUvsGPU","enPFCluster_CPUvsGPU",50,0,500,50,0,500);

  TH1F *deltaSumSeed  = new TH1F("deltaSumSeed", "sumSeed_{GPU} - sumSeed_{CPU}", 201, -100.5, 100.5);
  TH1F *deltaRH  = new TH1F("deltaRH", "nRH_{GPU} - nRH_{CPU}", 41, -20.5, 20.5);
  TH1F *deltaEn  = new TH1F("deltaEn", "E_{GPU} - E_{CPU}", 200, -10, 10);
  TH1F *deltaEta = new TH1F("deltaEta", "#eta_{GPU} - #eta_{CPU}", 200, -0.2, 0.2);
  TH1F *deltaPhi = new TH1F("deltaPhi", "#phi_{GPU} - #phi_{CPU}", 200, -0.2, 0.2);

  TH2F *coordinate = new TH2F("coordinate","coordinate",100,-3,3,100,-3.1415926,3.14159);
  TH1F *layer = new TH1F("layer","layer",7,0,7);

  TH1F *hTimers = new TH1F("timers", "GPU kernel timers (Event > 9)", 9, -0.5, 8.5);
  std::array<float,9> GPU_timers;
  Int_t numEvents = 0;
  Int_t topoIter = 0;
  Int_t nEdges = 0;
  Int_t nFracs = 0;

  Int_t nRHperPFCTotal_CPU = 0;
  Int_t nRHperPFCTotal_GPU = 0;


private:
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  
  //bool doComparison=true;
  bool doComparison=false;

  // inputs
  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;
  
  cms::cuda::ContextState cudaState_;
  cudaStream_t cudaStream = 0;

  PFClustering::common::CudaECALConstants cudaConstants;
  PFClustering::ECAL::ConfigurationParameters cudaConfig_;
  PFClustering::ECAL::InputDataCPU inputCPU;
  PFClustering::ECAL::OutputDataCPU outputCPU;
  PFClustering::ECAL::InputDataGPU inputGPU;
  PFClustering::ECAL::OutputDataGPU outputGPU;
  PFClustering::ECAL::ScratchDataGPU scratchGPU;

  std::unique_ptr<reco::PFClusterCollection> pfClustersFromCuda;
};

DEFINE_FWK_MODULE(PFClusterProducerCudaECAL);

#endif
