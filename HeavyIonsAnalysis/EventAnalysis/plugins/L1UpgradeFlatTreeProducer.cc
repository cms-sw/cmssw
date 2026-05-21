/*

derived from L1UpgradeTreeProducer.cc 
https://github.com/cms-sw/cmssw/blob/71ca6cc2c05f6ac6aa84b51976d1a06c9b6719a4/L1Trigger/L1TNtuples/plugins/L1UpgradeTreeProducer.cc

writes the same information as L1UpgradeTreeProducer, but using a simpler format.
L1UpgradeTreeProducer uses a format which requires L1AnalysisL1UpgradeDataFormat object to read the branches
This class writes information using standard data types which does not complicate branch reading.
 
*/

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
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1Upgrade.h"

//
// class declaration
//

class L1UpgradeFlatTreeProducer : public edm::one::EDAnalyzer<> {
public:
  explicit L1UpgradeFlatTreeProducer(const edm::ParameterSet&);
  ~L1UpgradeFlatTreeProducer() override;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

public:
  L1Analysis::L1AnalysisL1Upgrade* l1Upgrade;
  L1Analysis::L1AnalysisL1UpgradeDataFormat* l1UpgradeData;

private:
  unsigned maxL1Upgrade_;

  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // EDM input tags
  edm::EDGetTokenT<l1t::EGammaBxCollection> egToken_;
  std::vector<edm::EDGetTokenT<l1t::TauBxCollection>> tauTokens_;
  edm::EDGetTokenT<l1t::JetBxCollection> jetToken_;
  edm::EDGetTokenT<l1t::EtSumBxCollection> sumToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> muonToken_;

  // switches
  bool doEg_;
  bool doTau_;
  bool doJet_;
  bool doSum_;
  bool doMuon_;
};

L1UpgradeFlatTreeProducer::L1UpgradeFlatTreeProducer(const edm::ParameterSet& iConfig) {
  egToken_ = consumes<l1t::EGammaBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("egToken"));
  //tauToken_ = consumes<l1t::TauBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tauToken"));
  jetToken_ = consumes<l1t::JetBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("jetToken"));
  sumToken_ = consumes<l1t::EtSumBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("sumToken"));
  muonToken_ = consumes<l1t::MuonBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("muonToken"));

  const auto& taus = iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("tauTokens");
  for (const auto& tau : taus) {
    tauTokens_.push_back(consumes<l1t::TauBxCollection>(tau));
  }

  doEg_ = iConfig.getParameter<bool>("doEg");
  doTau_ = iConfig.getParameter<bool>("doTau");
  doJet_ = iConfig.getParameter<bool>("doJet");
  doSum_ = iConfig.getParameter<bool>("doSum");
  doMuon_ = iConfig.getParameter<bool>("doMuon");

  maxL1Upgrade_ = iConfig.getParameter<unsigned int>("maxL1Upgrade");

  l1Upgrade = new L1Analysis::L1AnalysisL1Upgrade();
  l1UpgradeData = l1Upgrade->getData();

  // set up output
  tree_ = fs_->make<TTree>("L1UpgradeFlatTree", "L1 objects");
  //tree_->Branch("L1Upgrade", "L1Analysis::L1AnalysisL1UpgradeDataFormat", &l1UpgradeData, 32000, 3);

  if (doEg_) {
    tree_->Branch("nEGs", &l1UpgradeData->nEGs);
    tree_->Branch("egEt", &l1UpgradeData->egEt);
    tree_->Branch("egEta", &l1UpgradeData->egEta);
    tree_->Branch("egPhi", &l1UpgradeData->egPhi);
    tree_->Branch("egIEt", &l1UpgradeData->egIEt);
    tree_->Branch("egIEta", &l1UpgradeData->egIEta);
    tree_->Branch("egIPhi", &l1UpgradeData->egIPhi);
    tree_->Branch("egIso", &l1UpgradeData->egIso);
    tree_->Branch("egBx", &l1UpgradeData->egBx);
    tree_->Branch("egTowerIPhi", &l1UpgradeData->egTowerIPhi);
    tree_->Branch("egTowerIEta", &l1UpgradeData->egTowerIEta);
    tree_->Branch("egRawEt", &l1UpgradeData->egRawEt);
    tree_->Branch("egIsoEt", &l1UpgradeData->egIsoEt);
    tree_->Branch("egFootprintEt", &l1UpgradeData->egFootprintEt);
    tree_->Branch("egNTT", &l1UpgradeData->egNTT);
    tree_->Branch("egShape", &l1UpgradeData->egShape);
    tree_->Branch("egTowerHoE", &l1UpgradeData->egTowerHoE);
    tree_->Branch("egHwQual", &l1UpgradeData->egHwQual);
  }

  if (doTau_) {
    tree_->Branch("nTaus", &l1UpgradeData->nTaus);
    tree_->Branch("tauEt", &l1UpgradeData->tauEt);
    tree_->Branch("tauEta", &l1UpgradeData->tauEta);
    tree_->Branch("tauPhi", &l1UpgradeData->tauPhi);
    tree_->Branch("tauIEt", &l1UpgradeData->tauIEt);
    tree_->Branch("tauIEta", &l1UpgradeData->tauIEta);
    tree_->Branch("tauIPhi", &l1UpgradeData->tauIPhi);
    tree_->Branch("tauIso", &l1UpgradeData->tauIso);
    tree_->Branch("tauBx", &l1UpgradeData->tauBx);
    tree_->Branch("tauTowerIPhi", &l1UpgradeData->tauTowerIPhi);
    tree_->Branch("tauTowerIEta", &l1UpgradeData->tauTowerIEta);
    tree_->Branch("tauRawEt", &l1UpgradeData->tauRawEt);
    tree_->Branch("tauIsoEt", &l1UpgradeData->tauIsoEt);
    tree_->Branch("tauNTT", &l1UpgradeData->tauNTT);
    tree_->Branch("tauHasEM", &l1UpgradeData->tauHasEM);
    tree_->Branch("tauIsMerged", &l1UpgradeData->tauIsMerged);
    tree_->Branch("tauHwQual", &l1UpgradeData->tauHwQual);
  }

  if (doJet_) {
    tree_->Branch("nJets", &l1UpgradeData->nJets);
    tree_->Branch("jetEt", &l1UpgradeData->jetEt);
    tree_->Branch("jetEta", &l1UpgradeData->jetEta);
    tree_->Branch("jetPhi", &l1UpgradeData->jetPhi);
    tree_->Branch("jetIEt", &l1UpgradeData->jetIEt);
    tree_->Branch("jetIEta", &l1UpgradeData->jetIEta);
    tree_->Branch("jetIPhi", &l1UpgradeData->jetIPhi);
    tree_->Branch("jetBx", &l1UpgradeData->jetBx);
    tree_->Branch("jetTowerIPhi", &l1UpgradeData->jetTowerIPhi);
    tree_->Branch("jetTowerIEta", &l1UpgradeData->jetTowerIEta);
    tree_->Branch("jetRawEt", &l1UpgradeData->jetRawEt);
    tree_->Branch("jetSeedEt", &l1UpgradeData->jetSeedEt);
    tree_->Branch("jetPUEt", &l1UpgradeData->jetPUEt);
    tree_->Branch("jetPUDonutEt0", &l1UpgradeData->jetPUDonutEt0);
    tree_->Branch("jetPUDonutEt1", &l1UpgradeData->jetPUDonutEt1);
    tree_->Branch("jetPUDonutEt2", &l1UpgradeData->jetPUDonutEt2);
    tree_->Branch("jetPUDonutEt3", &l1UpgradeData->jetPUDonutEt3);
  }

  if (doMuon_) {
    tree_->Branch("nMuons", &l1UpgradeData->nMuons);
    tree_->Branch("muonEt", &l1UpgradeData->muonEt);
    tree_->Branch("muonEta", &l1UpgradeData->muonEta);
    tree_->Branch("muonPhi", &l1UpgradeData->muonPhi);
    tree_->Branch("muonEtaAtVtx", &l1UpgradeData->muonEtaAtVtx);
    tree_->Branch("muonPhiAtVtx", &l1UpgradeData->muonPhiAtVtx);
    tree_->Branch("muonIEt", &l1UpgradeData->muonIEt);
    tree_->Branch("muonIEta", &l1UpgradeData->muonIEta);
    tree_->Branch("muonIPhi", &l1UpgradeData->muonIPhi);
    tree_->Branch("muonIEtaAtVtx", &l1UpgradeData->muonIEtaAtVtx);
    tree_->Branch("muonIPhiAtVtx", &l1UpgradeData->muonIPhiAtVtx);
    tree_->Branch("muonIDEta", &l1UpgradeData->muonIDEta);
    tree_->Branch("muonIDPhi", &l1UpgradeData->muonIDPhi);
    tree_->Branch("muonChg", &l1UpgradeData->muonChg);
    tree_->Branch("muonIso", &l1UpgradeData->muonIso);
    tree_->Branch("muonQual", &l1UpgradeData->muonQual);
    tree_->Branch("muonTfMuonIdx", &l1UpgradeData->muonTfMuonIdx);
    tree_->Branch("muonBx", &l1UpgradeData->muonBx);
  }

  if (doSum_) {
    tree_->Branch("nSums", &l1UpgradeData->nSums);
    tree_->Branch("sumType", &l1UpgradeData->sumType);
    tree_->Branch("sumEt", &l1UpgradeData->sumEt);
    tree_->Branch("sumPhi", &l1UpgradeData->sumPhi);
    tree_->Branch("sumIEt", &l1UpgradeData->sumIEt);
    tree_->Branch("sumIPhi", &l1UpgradeData->sumIPhi);
    tree_->Branch("sumBx", &l1UpgradeData->sumBx);
  }
}

L1UpgradeFlatTreeProducer::~L1UpgradeFlatTreeProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1UpgradeFlatTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  l1Upgrade->Reset();

  edm::Handle<l1t::EGammaBxCollection> eg;
  edm::Handle<l1t::JetBxCollection> jet;
  edm::Handle<l1t::EtSumBxCollection> sums;
  edm::Handle<l1t::MuonBxCollection> muon;

  if (doEg_)
    iEvent.getByToken(egToken_, eg);
  if (doJet_)
    iEvent.getByToken(jetToken_, jet);
  if (doSum_)
    iEvent.getByToken(sumToken_, sums);
  if (doMuon_)
    iEvent.getByToken(muonToken_, muon);

  if (doEg_) {
    if (eg.isValid()) {
      l1Upgrade->SetEm(eg, maxL1Upgrade_);
    } else {
      edm::LogWarning("MissingProduct") << "L1Upgrade Em not found. Branch will not be filled" << std::endl;
    }
  }

  if (doJet_) {
    if (jet.isValid()) {
      l1Upgrade->SetJet(jet, maxL1Upgrade_);
    } else {
      edm::LogWarning("MissingProduct") << "L1Upgrade Jets not found. Branch will not be filled" << std::endl;
    }
  }

  if (doSum_) {
    if (sums.isValid()) {
      l1Upgrade->SetSum(sums, maxL1Upgrade_);
    } else {
      edm::LogWarning("MissingProduct") << "L1Upgrade EtSums not found. Branch will not be filled" << std::endl;
    }
  }

  if (doMuon_) {
    if (muon.isValid()) {
      l1Upgrade->SetMuon(muon, maxL1Upgrade_);
    } else {
      edm::LogWarning("MissingProduct") << "L1Upgrade Muons not found. Branch will not be filled" << std::endl;
    }
  }

  if (doTau_) {
    for (auto& tautoken : tauTokens_) {
      edm::Handle<l1t::TauBxCollection> tau;
      iEvent.getByToken(tautoken, tau);
      if (tau.isValid()) {
        l1Upgrade->SetTau(tau, maxL1Upgrade_);
      } else {
        edm::LogWarning("MissingProduct") << "L1Upgrade Tau not found. Branch will not be filled" << std::endl;
      }
    }
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1UpgradeFlatTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1UpgradeFlatTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1UpgradeFlatTreeProducer);
