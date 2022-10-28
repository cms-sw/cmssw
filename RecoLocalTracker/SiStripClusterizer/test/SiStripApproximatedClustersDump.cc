// system includes
#include <memory>
#include <iostream>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//ROOT inclusion
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TMath.h"
#include "TList.h"
#include "TString.h"

//
// class decleration
//

class SiStripApproximatedClustersDump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiStripApproximatedClustersDump(const edm::ParameterSet&);
  ~SiStripApproximatedClustersDump() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::InputTag inputTagClusters;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripApproximateCluster>> clusterToken;

  TTree* outNtuple;
  edm::Service<TFileService> fs;

  uint32_t detId;
  uint16_t barycenter;
  uint16_t width;
  uint8_t avCharge;
  edm::EventNumber_t eventN;
};

SiStripApproximatedClustersDump::SiStripApproximatedClustersDump(const edm::ParameterSet& conf) {
  inputTagClusters = conf.getParameter<edm::InputTag>("approxSiStripClustersTag");
  clusterToken = consumes<edmNew::DetSetVector<SiStripApproximateCluster>>(inputTagClusters);

  usesResource("TFileService");

  outNtuple = fs->make<TTree>("ApproxClusters", "ApproxClusters");
  outNtuple->Branch("event", &eventN, "event/i");
  outNtuple->Branch("detId", &detId, "detId/i");
  outNtuple->Branch("barycenter", &barycenter, "barycenter/F");
  outNtuple->Branch("width", &width, "width/b");
  outNtuple->Branch("charge", &avCharge, "charge/b");
}

SiStripApproximatedClustersDump::~SiStripApproximatedClustersDump() = default;

void SiStripApproximatedClustersDump::analyze(const edm::Event& event, const edm::EventSetup& es) {
  edm::Handle<edmNew::DetSetVector<SiStripApproximateCluster>> clusterCollection = event.getHandle(clusterToken);

  for (const auto& detClusters : *clusterCollection) {
    detId = detClusters.detId();
    eventN = event.id().event();

    for (const auto& cluster : detClusters) {
      barycenter = cluster.barycenter();
      width = cluster.width();
      avCharge = cluster.avgCharge();
      outNtuple->Fill();
    }
  }
}

void SiStripApproximatedClustersDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("approxSiStripClustersTag", edm::InputTag("SiStripClusters2ApproxClusters"));
  descriptions.add("SiStripApproximatedClustersDump", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripApproximatedClustersDump);
