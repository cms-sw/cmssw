// -*- C++ -*-
//
// Package:    L1TriggerDPG/L1Ntuples
// Class:      L1UpgradeTreeProducer
//
/**\class L1UpgradeTreeProducer L1UpgradeTreeProducer.cc L1TriggerDPG/L1Ntuples/src/L1UpgradeTreeProducer.cc

Description: Produce L1 Extra tree

Implementation:

*/
//
// Original Author:
//         Created:
// $Id: L1UpgradeTreeProducer.cc,v 1.8 2012/08/29 12:44:03 jbrooke Exp $
//
//

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
#include "DataFormats/L1Trigger/interface/MuonShower.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1Upgrade.h"

//
// class declaration
//

class L1UpgradeTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1UpgradeTreeProducer(const edm::ParameterSet&);
  ~L1UpgradeTreeProducer() override = default;

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
  const edm::EDGetTokenT<l1t::EGammaBxCollection> egToken_;
  const edm::EDGetTokenT<l1t::JetBxCollection> jetToken_;
  const edm::EDGetTokenT<l1t::EtSumBxCollection> sumToken_;
  const edm::EDGetTokenT<l1t::MuonBxCollection> muonToken_;
  const edm::EDGetTokenT<l1t::MuonShowerBxCollection> muonShowerToken_;

  const edm::EDGetTokenT<l1t::EtSumBxCollection> sumZDCToken_;

  std::vector<edm::EDGetTokenT<l1t::TauBxCollection>> tauTokens_;
};

L1UpgradeTreeProducer::L1UpgradeTreeProducer(const edm::ParameterSet& iConfig)
    : egToken_(consumes<l1t::EGammaBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("egToken"))),
      jetToken_(consumes<l1t::JetBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("jetToken"))),
      sumToken_(consumes<l1t::EtSumBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("sumToken"))),
      muonToken_(consumes<l1t::MuonBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("muonToken"))),
      muonShowerToken_(
          consumes<l1t::MuonShowerBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("muonShowerToken"))),
      sumZDCToken_(consumes<l1t::EtSumBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("sumZDCToken"))) {
  const auto& taus = iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("tauTokens");
  for (const auto& tau : taus) {
    tauTokens_.push_back(consumes<l1t::TauBxCollection>(tau));
  }

  maxL1Upgrade_ = iConfig.getParameter<unsigned int>("maxL1Upgrade");

  l1Upgrade = new L1Analysis::L1AnalysisL1Upgrade();
  l1UpgradeData = l1Upgrade->getData();

  usesResource(TFileService::kSharedResource);

  // set up output
  tree_ = fs_->make<TTree>("L1UpgradeTree", "L1UpgradeTree");
  tree_->Branch("L1Upgrade", "L1Analysis::L1AnalysisL1UpgradeDataFormat", &l1UpgradeData, 32000, 3);
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1UpgradeTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  l1Upgrade->Reset();

  edm::Handle<l1t::EGammaBxCollection> eg;
  edm::Handle<l1t::JetBxCollection> jet;
  edm::Handle<l1t::EtSumBxCollection> sums;
  edm::Handle<l1t::MuonBxCollection> muon;
  edm::Handle<l1t::MuonShowerBxCollection> muonShower;
  edm::Handle<l1t::EtSumBxCollection> sumsZDC;

  iEvent.getByToken(egToken_, eg);
  iEvent.getByToken(jetToken_, jet);
  iEvent.getByToken(sumToken_, sums);
  iEvent.getByToken(muonToken_, muon);
  iEvent.getByToken(muonShowerToken_, muonShower);
  iEvent.getByToken(sumZDCToken_, sumsZDC);

  if (eg.isValid()) {
    l1Upgrade->SetEm(eg, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade Em not found. Branch will not be filled" << std::endl;
  }
  if (jet.isValid()) {
    l1Upgrade->SetJet(jet, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade Jets not found. Branch will not be filled" << std::endl;
  }

  if (sums.isValid()) {
    l1Upgrade->SetSum(sums, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade EtSums not found. Branch will not be filled" << std::endl;
  }

  if (muon.isValid()) {
    l1Upgrade->SetMuon(muon, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade Muons not found. Branch will not be filled" << std::endl;
  }

  if (muonShower.isValid()) {
    l1Upgrade->SetMuonShower(muonShower, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade Muon Showers not found. Branch will not be filled" << std::endl;
  }

  if (sumsZDC.isValid()) {
    l1Upgrade->SetSumZDC(sumsZDC, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade EtZDCSums not found. Branch will not be filled" << std::endl;
  }

  for (auto& tautoken : tauTokens_) {
    edm::Handle<l1t::TauBxCollection> tau;
    iEvent.getByToken(tautoken, tau);
    if (tau.isValid()) {
      l1Upgrade->SetTau(tau, maxL1Upgrade_);
    } else {
      edm::LogWarning("MissingProduct") << "L1Upgrade Tau not found. Branch will not be filled" << std::endl;
    }
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1UpgradeTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1UpgradeTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1UpgradeTreeProducer);
