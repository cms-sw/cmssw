// -*- C++ -*-
//
// Package:    L1Trigger/L1TNtuples
// Class:      L1JetRecoTreeProducer
//
/**\class L1JetRecoTreeProducer L1JetRecoTreeProducer.cc L1Trigger/L1TNtuples/src/L1JetRecoTreeProducer.cc

 Description: Produces tree containing reco quantities


*/


// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// cond formats
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

// data formats
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/JetID.h"


// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoJet.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMet.h"

//
// class declaration
//

class L1JetRecoTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1JetRecoTreeProducer(const edm::ParameterSet&);
  ~L1JetRecoTreeProducer();


private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

public:
  L1Analysis::L1AnalysisRecoJet*        jet;

  L1Analysis::L1AnalysisRecoJetDataFormat*              jet_data;

private:

  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree * tree_;

  // EDM input tags
  edm::InputTag caloJetTag_;
  edm::InputTag caloJetIdTag_;
  edm::EDGetTokenT<reco::JetCorrector> jetCorrectorToken_;

  // debug stuff
  bool caloJetsMissing_;
  double jetptThreshold_;
  unsigned int maxCl_;
  unsigned int maxJet_;
  unsigned int maxVtx_;
  unsigned int maxTrk_;
};



L1JetRecoTreeProducer::L1JetRecoTreeProducer(const edm::ParameterSet& iConfig):
  caloJetTag_(iConfig.getUntrackedParameter("caloJetTag",edm::InputTag("ak4CaloJets"))),
  caloJetIdTag_(iConfig.getUntrackedParameter("jetIdTag",edm::InputTag("ak4JetID"))),
  caloJetsMissing_(false)
{

  jetCorrectorToken_ = consumes<reco::JetCorrector>(iConfig.getUntrackedParameter<edm::InputTag>("jetCorrToken"));

  jetptThreshold_ = iConfig.getParameter<double>      ("jetptThreshold");
  maxJet_         = iConfig.getParameter<unsigned int>("maxJet");

  jet           = new L1Analysis::L1AnalysisRecoJet();
  jet_data           = jet->getData();

  // set up output
  tree_=fs_->make<TTree>("JetRecoTree", "JetRecoTree");
  tree_->Branch("Jet",           "L1Analysis::L1AnalysisRecoJetDataFormat",         &jet_data,                32000, 3);

}


L1JetRecoTreeProducer::~L1JetRecoTreeProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void L1JetRecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  jet->Reset();

  // get jets  & co...
  edm::Handle<reco::CaloJetCollection> recoCaloJets;
  edm::Handle<edm::ValueMap<reco::JetID> > jetsID;
  edm::Handle<reco::JetCorrector> jetCorr;

  iEvent.getByLabel(caloJetTag_, recoCaloJets);
  //iEvent.getByLabel(jetIdTag_,jetsID);
  //iEvent.getByToken(jetCorrectorToken_, jetCorr);

  if (recoCaloJets.isValid()) {
    jet->SetCaloJet(iEvent, iSetup, recoCaloJets, maxJet_); //jetsID, maxJet_);
  }
  else {
    if (!caloJetsMissing_) {edm::LogWarning("MissingProduct") << caloJetTag_ << " : CaloJets not found.  Branch will not be filled" << std::endl;}
    caloJetsMissing_ = true;
  }

  tree_->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void
L1JetRecoTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1JetRecoTreeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1JetRecoTreeProducer);
