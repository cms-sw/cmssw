// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// data formats
#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeTfMuonShower.h"

//
// class declaration
//

class L1UpgradeTfMuonShowerTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1UpgradeTfMuonShowerTreeProducer(const edm::ParameterSet&);
  ~L1UpgradeTfMuonShowerTreeProducer() override = default;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

public:
  L1Analysis::L1AnalysisL1UpgradeTfMuonShower l1UpgradeEmtf;
  L1Analysis::L1AnalysisL1UpgradeTfMuonShowerDataFormat* l1UpgradeEmtfData;

private:
  unsigned maxL1UpgradeTfMuonShower_;

  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // EDM input tags
  edm::EDGetTokenT<l1t::RegionalMuonShowerBxCollection> emtfMuonShowerToken_;
};

L1UpgradeTfMuonShowerTreeProducer::L1UpgradeTfMuonShowerTreeProducer(const edm::ParameterSet& iConfig) {
  emtfMuonShowerToken_ = consumes<l1t::RegionalMuonShowerBxCollection>(
      iConfig.getUntrackedParameter<edm::InputTag>("emtfMuonShowerToken"));

  maxL1UpgradeTfMuonShower_ = iConfig.getParameter<unsigned int>("maxL1UpgradeTfMuonShower");

  l1UpgradeEmtfData = l1UpgradeEmtf.getData();

  usesResource(TFileService::kSharedResource);

  // set up output
  tree_ = fs_->make<TTree>("L1UpgradeTfMuonShowerTree", "L1UpgradeTfMuonShowerTree");
  tree_->Branch(
      "L1UpgradeEmtfMuonShower", "L1Analysis::L1AnalysisL1UpgradeTfMuonShowerDataFormat", &l1UpgradeEmtfData, 32000, 3);
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1UpgradeTfMuonShowerTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  l1UpgradeEmtf.Reset();

  edm::Handle<l1t::RegionalMuonShowerBxCollection> emtfMuonShower;

  iEvent.getByToken(emtfMuonShowerToken_, emtfMuonShower);

  if (emtfMuonShower.isValid()) {
    l1UpgradeEmtf.SetTfMuonShower(*emtfMuonShower, maxL1UpgradeTfMuonShower_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade EMTF muons not found. Branch will not be filled" << std::endl;
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1UpgradeTfMuonShowerTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1UpgradeTfMuonShowerTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1UpgradeTfMuonShowerTreeProducer);
