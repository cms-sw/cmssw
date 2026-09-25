//============================================================================================================================
// Class:      HGCALGPUvsCPUComparisonHists                     --------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------
/**\class HGCALGPUvsCPUComparisonHists HGCALGPUvsCPUComparisonHists.cc DQM/HGCAL/plugins/HGCALGPUvsCPUComparisonHists.cc
------------------------------------------------------------------------------------------------------------------------------
 Description: This class produces histograms to compare GPU- and CPU-based HGCAL reconstruction  ---
 -----------------------------------------------------------------------------------------------------------------------------
 Implementation:                                                                          ---
     This DQMEDAnalyzer is meant to be used with CMSSW >= 16_1_0                          ---
*/
//========================================================================================
// Authors:  Fabio Iemmi (IHEP)                                      ---------------------
//         Created:  TUE, 28 Apr 2026 16:05:28 GMT  --------------------------------------
//========================================================================================

#include <string>
#include <unordered_map>
#include <utility>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class HGCALGPUvsCPUComparisonHists : public DQMEDAnalyzer {
public:
  explicit HGCALGPUvsCPUComparisonHists(const edm::ParameterSet&);
  ~HGCALGPUvsCPUComparisonHists() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob(const edm::EventSetup& iSetup);
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;

private:
  const edm::EDGetTokenT<reco::CaloClusterCollection> tokenMonitoredLayerClusters_;
  const edm::EDGetTokenT<reco::CaloClusterCollection> tokenReferenceLayerClusters_;
  const std::string topFolderName_;
  MonitorElement* hLayerCluster_x;
  MonitorElement* hLayerCluster_y;
  MonitorElement* hLayerCluster_z;
  MonitorElement* hLayerCluster_eta;
  MonitorElement* hLayerCluster_phi;
  MonitorElement* hLayerCluster_e;
  MonitorElement* hLayerCluster_nRecHits;
  MonitorElement* hLayerCluster2D_x;
  MonitorElement* hLayerCluster2D_y;
  MonitorElement* hLayerCluster2D_z;
  MonitorElement* hLayerCluster2D_eta;
  MonitorElement* hLayerCluster2D_phi;
  MonitorElement* hLayerCluster2D_e;
  MonitorElement* hLayerCluster2D_nRecHits;
};

HGCALGPUvsCPUComparisonHists::HGCALGPUvsCPUComparisonHists(const edm::ParameterSet& iConfig)
    : tokenMonitoredLayerClusters_(
          consumes<reco::CaloClusterCollection>(iConfig.getParameter<edm::InputTag>("monitoredLayerClusters"))),
      tokenReferenceLayerClusters_(
          consumes<reco::CaloClusterCollection>(iConfig.getParameter<edm::InputTag>("referenceLayerClusters"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")) {}

void HGCALGPUvsCPUComparisonHists::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("monitoredLayerClusters", edm::InputTag("hltMergeLayerClusters"));
  desc.add<edm::InputTag>("referenceLayerClusters", edm::InputTag("hltMergeLayerClustersSerialSync"));
  desc.add<std::string>("topFolderName", "HLT/HeterogeneousComparisons/HGCalMonitoring");
  descriptions.addWithDefaultLabel(desc);
}

void HGCALGPUvsCPUComparisonHists::beginJob(const edm::EventSetup& iSetup) {}

void HGCALGPUvsCPUComparisonHists::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(topFolderName_);
  //For a given variable x, 1D plots store Delta(x), 2D plots show x_GPU vs x_CPU
  //1D
  hLayerCluster_x = iBooker.book1D("hLayerCluster_x", "hLayerCluster_x", 100, -0.01, 0.01);
  hLayerCluster_y = iBooker.book1D("hLayerCluster_y", "hLayerCluster_y", 100, -0.01, 0.01);
  hLayerCluster_z = iBooker.book1D("hLayerCluster_z", "hLayerCluster_z", 100, -0.01, 0.01);
  hLayerCluster_eta = iBooker.book1D("hLayerCluster_eta", "hLayerCluster_eta", 100, -0.01, 0.01);
  hLayerCluster_phi = iBooker.book1D("hLayerCluster_phi", "hLayerCluster_phi", 100, -0.01, 0.01);
  hLayerCluster_e = iBooker.book1D("hLayerCluster_e", "hLayerCluster_e", 100, -0.01, 0.01);
  hLayerCluster_nRecHits = iBooker.book1D("hLayerCluster_nRecHits", "hLayerCluster_nRecHits", 100, -0.01, 0.01);
  //2D
  hLayerCluster2D_x = iBooker.book2D("hLayerCluster2D_x", "hLayerCluster2D_x", 100, -50, 50, 100, -50, 50);
  hLayerCluster2D_y = iBooker.book2D("hLayerCluster2D_y", "hLayerCluster2D_y", 100, -50, 50, 100, -50, 50);
  hLayerCluster2D_z = iBooker.book2D("hLayerCluster2D_z", "hLayerCluster2D_z", 100, -500, 500, 100, -500, 500);
  hLayerCluster2D_eta = iBooker.book2D("hLayerCluster2D_eta", "hLayerCluster2D_eta", 100, -3.5, 3.5, 100, -3.5, 3.5);
  hLayerCluster2D_phi = iBooker.book2D("hLayerCluster2D_phi", "hLayerCluster2D_phi", 100, -3.5, 3.5, 100, -3.5, 3.5);
  hLayerCluster2D_e = iBooker.book2D("hLayerCluster2D_e", "hLayerCluster2D_e", 100, 0, 30, 100, 0, 30);
  hLayerCluster2D_nRecHits =
      iBooker.book2D("hLayerCluster2D_nRecHits", "hLayerCluster2D_nRecHits", 60, 0, 60, 60, 0, 60);
}

void HGCALGPUvsCPUComparisonHists::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Get monitored (GPU) and reference (CPU) LayerCluster collections
  const auto& monitoredHandle = iEvent.getHandle(tokenMonitoredLayerClusters_);
  const auto& referenceHandle = iEvent.getHandle(tokenReferenceLayerClusters_);
  if (!monitoredHandle.isValid() || !referenceHandle.isValid()) {
    edm::LogWarning("HGCALGPUvsCPUComparisonHists") << "Monitored or reference LayerCluster collection is invalid.";
    return;
  }
  const reco::CaloClusterCollection& monitoredLayerClusters = *monitoredHandle;
  const reco::CaloClusterCollection& referenceLayerClusters = *referenceHandle;

  //look for GPU and CPU LayerClusters whose seeds match
  //map LC seeds to LC indices for the reference collection
  std::unordered_map<uint32_t, std::pair<unsigned, bool>>
      seedToIdx;  //map seed of reference LC to index and wether or not it matches a monitored LC
  seedToIdx.reserve(referenceLayerClusters.size());
  for (unsigned idx = 0; idx < referenceLayerClusters.size(); idx++) {
    auto [it, inserted] = seedToIdx.try_emplace(
        referenceLayerClusters[idx].seed(), idx, false);  //initialze all reference LCs as unmatched
    if (!inserted) {
      edm::LogWarning("HGCALGPUvsCPUComparisonHists") << "Duplicate seed in reference collection.";
      continue;
    }
  }
  //look for matches in the monitored collection and, if any, fill histograms
  for (unsigned i = 0; i < monitoredLayerClusters.size(); i++) {
    const auto& monitored = monitoredLayerClusters[i];
    auto it = seedToIdx.find(monitored.seed());
    if (it != seedToIdx.end() && it->second.second == false) {
      it->second.second = true;  //establish a match
      const auto& reference = referenceLayerClusters[it->second.first];

      hLayerCluster_x->Fill(monitored.x() - reference.x());
      hLayerCluster_y->Fill(monitored.y() - reference.y());
      hLayerCluster_z->Fill(monitored.z() - reference.z());
      hLayerCluster_eta->Fill(monitored.eta() - reference.eta());
      hLayerCluster_phi->Fill(monitored.phi() - reference.phi());
      hLayerCluster_e->Fill(monitored.energy() - reference.energy());
      hLayerCluster_nRecHits->Fill(monitored.size() - reference.size());

      hLayerCluster2D_x->Fill(reference.x(), monitored.x());
      hLayerCluster2D_y->Fill(reference.y(), monitored.y());
      hLayerCluster2D_z->Fill(reference.z(), monitored.z());
      hLayerCluster2D_eta->Fill(reference.eta(), monitored.eta());
      hLayerCluster2D_phi->Fill(reference.phi(), monitored.phi());
      hLayerCluster2D_e->Fill(reference.energy(), monitored.energy());
      hLayerCluster2D_nRecHits->Fill(reference.size(), monitored.size());
    } else {
      edm::LogWarning("HGCALGPUvsCPUComparisonHists") << "No match or duplicate match to reference collection found.";
      continue;
    }
  }
}

DEFINE_FWK_MODULE(HGCALGPUvsCPUComparisonHists);
