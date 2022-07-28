// -*- C++ -*-
//
// Package:    L1Trigger/L1TNtuples
// Class:      L1ExtraTreeProducer
//
/**\class L1ExtraTreeProducer L1ExtraTreeProducer.cc L1Trigger/L1TNtuples/src/L1ExtraTreeProducer.cc

Description: Produce L1 Extra tree

Implementation:
     
*/
//
// Original Author:  Alex Tapper
//         Created:
// $Id: L1ExtraTreeProducer.cc,v 1.8 2012/08/29 12:44:03 jbrooke Exp $
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
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1Extra.h"

//
// class declaration
//

class L1ExtraTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1ExtraTreeProducer(const edm::ParameterSet&);
  ~L1ExtraTreeProducer() override;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

public:
  L1Analysis::L1AnalysisL1Extra* l1Extra;
  L1Analysis::L1AnalysisL1ExtraDataFormat* l1ExtraData;

private:
  unsigned maxL1Extra_;

  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // EDM input tags
  edm::EDGetTokenT<l1extra::L1EmParticleCollection> nonIsoEmToken_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection> isoEmToken_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> tauJetToken_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> isoTauJetToken_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> cenJetToken_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> fwdJetToken_;
  edm::EDGetTokenT<l1extra::L1MuonParticleCollection> muonToken_;
  edm::EDGetTokenT<l1extra::L1EtMissParticleCollection> metToken_;
  edm::EDGetTokenT<l1extra::L1EtMissParticleCollection> mhtToken_;
  edm::EDGetTokenT<l1extra::L1HFRingsCollection> hfRingsToken_;
};

L1ExtraTreeProducer::L1ExtraTreeProducer(const edm::ParameterSet& iConfig) {
  nonIsoEmToken_ = consumes<l1extra::L1EmParticleCollection>(
      iConfig.getUntrackedParameter("nonIsoEmToken", edm::InputTag("l1extraParticles:NonIsolated")));
  isoEmToken_ = consumes<l1extra::L1EmParticleCollection>(
      iConfig.getUntrackedParameter("isoEmToken", edm::InputTag("l1extraParticles:Isolated")));
  tauJetToken_ = consumes<l1extra::L1JetParticleCollection>(
      iConfig.getUntrackedParameter("tauJetToken", edm::InputTag("l1extraParticles:Tau")));
  isoTauJetToken_ = consumes<l1extra::L1JetParticleCollection>(
      iConfig.getUntrackedParameter("isoTauJetToken", edm::InputTag("l1extraParticles:IsoTau")));
  cenJetToken_ = consumes<l1extra::L1JetParticleCollection>(
      iConfig.getUntrackedParameter("cenJetToken", edm::InputTag("l1extraParticles:Central")));
  fwdJetToken_ = consumes<l1extra::L1JetParticleCollection>(
      iConfig.getUntrackedParameter("fwdJetToken", edm::InputTag("l1extraParticles:Forward")));
  muonToken_ = consumes<l1extra::L1MuonParticleCollection>(
      iConfig.getUntrackedParameter("muonToken", edm::InputTag("l1extraParticles")));
  metToken_ = consumes<l1extra::L1EtMissParticleCollection>(
      iConfig.getUntrackedParameter("metToken", edm::InputTag("l1extraParticles:MET")));
  mhtToken_ = consumes<l1extra::L1EtMissParticleCollection>(
      iConfig.getUntrackedParameter("mhtToken", edm::InputTag("l1extraParticles:MHT")));
  hfRingsToken_ = consumes<l1extra::L1HFRingsCollection>(
      iConfig.getUntrackedParameter("hfRingsToken", edm::InputTag("l1extraParticles")));

  maxL1Extra_ = iConfig.getParameter<unsigned int>("maxL1Extra");

  l1Extra = new L1Analysis::L1AnalysisL1Extra();
  l1ExtraData = l1Extra->getData();

  usesResource(TFileService::kSharedResource);
  // set up output
  tree_ = fs_->make<TTree>("L1ExtraTree", "L1ExtraTree");
  tree_->Branch("L1Extra", "L1Analysis::L1AnalysisL1ExtraDataFormat", &l1ExtraData, 32000, 3);
}

L1ExtraTreeProducer::~L1ExtraTreeProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1ExtraTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  l1Extra->Reset();

  edm::Handle<l1extra::L1EmParticleCollection> isoEm;
  edm::Handle<l1extra::L1EmParticleCollection> nonIsoEm;
  edm::Handle<l1extra::L1JetParticleCollection> cenJet;
  edm::Handle<l1extra::L1JetParticleCollection> fwdJet;
  edm::Handle<l1extra::L1JetParticleCollection> tauJet;
  edm::Handle<l1extra::L1JetParticleCollection> isoTauJet;
  edm::Handle<l1extra::L1MuonParticleCollection> muon;
  ;
  edm::Handle<l1extra::L1EtMissParticleCollection> mets;
  edm::Handle<l1extra::L1EtMissParticleCollection> mhts;
  edm::Handle<l1extra::L1HFRingsCollection> hfRings;

  iEvent.getByToken(nonIsoEmToken_, nonIsoEm);
  iEvent.getByToken(isoEmToken_, isoEm);
  iEvent.getByToken(tauJetToken_, tauJet);
  iEvent.getByToken(isoTauJetToken_, isoTauJet);
  iEvent.getByToken(cenJetToken_, cenJet);
  iEvent.getByToken(fwdJetToken_, fwdJet);
  iEvent.getByToken(muonToken_, muon);
  iEvent.getByToken(metToken_, mets);
  iEvent.getByToken(mhtToken_, mhts);
  iEvent.getByToken(hfRingsToken_, hfRings);

  if (isoEm.isValid()) {
    l1Extra->SetIsoEm(isoEm, maxL1Extra_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra Iso Em not found. Branch will not be filled" << std::endl;
  }

  if (nonIsoEm.isValid()) {
    l1Extra->SetNonIsoEm(nonIsoEm, maxL1Extra_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra Non Iso Em not found. Branch will not be filled" << std::endl;
  }

  if (cenJet.isValid()) {
    l1Extra->SetCenJet(cenJet, maxL1Extra_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra Central Jets not found. Branch will not be filled" << std::endl;
  }

  if (tauJet.isValid()) {
    l1Extra->SetTauJet(tauJet, maxL1Extra_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra Tau Jets not found. Branch will not be filled" << std::endl;
  }

  if (isoTauJet.isValid()) {
    l1Extra->SetIsoTauJet(isoTauJet, maxL1Extra_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra Iso Tau Jets not found. Branch will not be filled" << std::endl;
  }

  if (fwdJet.isValid()) {
    l1Extra->SetFwdJet(fwdJet, maxL1Extra_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra Forward Jets not found. Branch will not be filled" << std::endl;
  }

  if (muon.isValid()) {
    l1Extra->SetMuon(muon, maxL1Extra_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra Muons not found. Branch will not be filled" << std::endl;
  }

  if (mets.isValid()) {
    l1Extra->SetMet(mets);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra MET not found. Branch will not be filled" << std::endl;
  }

  if (mhts.isValid()) {
    l1Extra->SetMht(mhts);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra MHT not found. Branch will not be filled" << std::endl;
  }

  if (hfRings.isValid()) {
    l1Extra->SetHFring(hfRings);
  } else {
    edm::LogWarning("MissingProduct") << "L1Extra HF Rings not found. Branch will not be filled" << std::endl;
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1ExtraTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1ExtraTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1ExtraTreeProducer);
