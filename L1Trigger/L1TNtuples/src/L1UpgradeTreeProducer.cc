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
#include "FWCore/Framework/interface/EDAnalyzer.h"
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

class L1UpgradeTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1UpgradeTreeProducer(const edm::ParameterSet&);
  ~L1UpgradeTreeProducer();
  
  
private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

public:
  
  L1Analysis::L1AnalysisL1Upgrade* l1Upgrade;
  L1Analysis::L1AnalysisL1UpgradeDataFormat * l1UpgradeData;

private:

  unsigned maxL1Upgrade_;

  // output file
  edm::Service<TFileService> fs_;
  
  // tree
  TTree * tree_;
 
  // EDM input tags
  edm::InputTag egLabel_;
  edm::InputTag tauLabel_;
  edm::InputTag jetLabel_;
  edm::InputTag sumLabel_;
  edm::InputTag muonLabel_;

};



L1UpgradeTreeProducer::L1UpgradeTreeProducer(const edm::ParameterSet& iConfig):
  egLabel_(iConfig.getUntrackedParameter("egLabel",edm::InputTag("caloStage2Digis"))),
  tauLabel_(iConfig.getUntrackedParameter("tauLabel",edm::InputTag("caloStage2Digis"))),
  jetLabel_(iConfig.getUntrackedParameter("jetLabel",edm::InputTag("caloStage2Digis"))),
  sumLabel_(iConfig.getUntrackedParameter("sumLabel",edm::InputTag("caloStage2Digis"))),
  muonLabel_(iConfig.getUntrackedParameter("muonLabel",edm::InputTag("")))
{
 
  maxL1Upgrade_ = iConfig.getParameter<unsigned int>("maxL1Upgrade");
 
  l1Upgrade     = new L1Analysis::L1AnalysisL1Upgrade();
  l1UpgradeData = l1Upgrade->getData();
  
  // set up output
  tree_=fs_->make<TTree>("L1UpgradeTree", "L1UpgradeTree");
  tree_->Branch("L1Upgrade", "L1Analysis::L1AnalysisL1UpgradeDataFormat", &l1UpgradeData, 32000, 3);

}


L1UpgradeTreeProducer::~L1UpgradeTreeProducer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1UpgradeTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  l1Upgrade->Reset();

  edm::Handle<l1t::EGammaBxCollection> eg;
  edm::Handle<l1t::TauBxCollection> tau;
  edm::Handle<l1t::JetBxCollection> jet;
  edm::Handle<l1t::EtSumBxCollection> sums;
  edm::Handle<l1t::MuonBxCollection> muon; ;

  iEvent.getByLabel(egLabel_,   eg);
  iEvent.getByLabel(tauLabel_,  tau);
  iEvent.getByLabel(jetLabel_,  jet);
  iEvent.getByLabel(sumLabel_, sums);
  iEvent.getByLabel(muonLabel_, muon);

  if (eg.isValid()){ 
    l1Upgrade->SetEm(eg, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade Em not found. Branch will not be filled" << std::endl;
  }

  if (tau.isValid()){ 
    l1Upgrade->SetTau(tau, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade Tau not found. Branch will not be filled" << std::endl;
  }

  if (jet.isValid()){ 
    l1Upgrade->SetJet(jet, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade Jets not found. Branch will not be filled" << std::endl;
  }

  if (sums.isValid()){ 
    l1Upgrade->SetSum(sums, maxL1Upgrade_);  
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade EtSums not found. Branch will not be filled" << std::endl;
  }

  if (muon.isValid()){ 
    l1Upgrade->SetMuon(muon, maxL1Upgrade_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade Muons not found. Branch will not be filled" << std::endl;
  }

  tree_->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void 
L1UpgradeTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1UpgradeTreeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1UpgradeTreeProducer);
