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


//taus
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

// #include "CommonTools/UtilAlgos/interface/PhysObjectMatcher.h"
// #include "CommonTools/UtilAlgos/interface/MCMatchSelector.h"
// #include "CommonTools/UtilAlgos/interface/DummyMatchSelector.h"
// #include "CommonTools/UtilAlgos/interface/MatchByDRDPt.h"
// #include "DataFormats/TauReco/interface/PFTauDecayMode.h"
// #include "DataFormats/TauReco/interface/PFTauDecayModeFwd.h"
// #include "DataFormats/TauReco/interface/PFTau.h"


// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoTau.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMet.h"

//
// class declaration
//

class L1TauRecoTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1TauRecoTreeProducer(const edm::ParameterSet&);
  ~L1TauRecoTreeProducer();


private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

public:
  L1Analysis::L1AnalysisRecoTau*        tau;

  L1Analysis::L1AnalysisRecoTauDataFormat*              tau_data;

private:

  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree * tree_;

  // EDM input tags
  edm::EDGetTokenT<reco::PFTauCollection>       TauToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator>    DMFindingToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator>    DMFindingOldToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator>    TightIsoToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator>    LooseIsoToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator>    LooseAntiMuonToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator>    TightAntiMuonToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator>    VLooseAntiElectronToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator>    LooseAntiElectronToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator>    TightAntiElectronToken_;

  // edm::EDGetTokenT<reco::CaloJetCollection>     caloJetToken_;
  // edm::EDGetTokenT<edm::ValueMap<reco::JetID> > caloJetIdToken_;
  // edm::EDGetTokenT<reco::JetCorrector>          jetCorrectorToken_;

  // debug stuff
  bool caloJetsMissing_;
  double jetptThreshold_;
  unsigned int maxCl_;
  std::string period_;
  unsigned int maxTau_;
  unsigned int maxVtx_;
  unsigned int maxTrk_;
};



L1TauRecoTreeProducer::L1TauRecoTreeProducer(const edm::ParameterSet& iConfig):
  caloJetsMissing_(false)
{

  period_ = iConfig.getParameter<std::string>("period");

  if(period_=="2015")
    {
      maxTau_         = iConfig.getParameter<unsigned int>("maxTau");
      TauToken_ = consumes<reco::PFTauCollection>(iConfig.getUntrackedParameter("TauToken",edm::InputTag("hpsPFTauProducer")));
      DMFindingToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("DMFindingToken",edm::InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs")));
      DMFindingOldToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("DMFindingOldToken",edm::InputTag("hpsPFTauDiscriminationByDecayModeFindingOldDMs")));
      TightIsoToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("TightIsoToken",edm::InputTag("hpsPFTauDiscriminationByTightIsolation")));
      LooseIsoToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("LooseIsoToken",edm::InputTag("hpsPFTauDiscriminationByLooseIsolation")));
      LooseAntiMuonToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("LooseAntiMuonToken",edm::InputTag("hpsPFTauDiscriminationByLooseMuonRejection")));
      TightAntiMuonToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("TightAntiMuonToken",edm::InputTag("hpsPFTauDiscriminationByTightMuonRejection")));
      VLooseAntiElectronToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("VLooseAntiElectronToken",edm::InputTag("hpsPFTauDiscriminationByMVA5VLooseElectronRejection")));
      LooseAntiElectronToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("LooseAntiElectronToken",edm::InputTag("hpsPFTauDiscriminationByMVA5LooseElectronRejection")));
      TightAntiElectronToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("TightAntiElectronToken",edm::InputTag("hpsPFTauDiscriminationByMVA5TightElectronRejection")));
    }
  else if(period_=="2016")
    {
      maxTau_         = iConfig.getParameter<unsigned int>("maxTau");
      TauToken_ = consumes<reco::PFTauCollection>(iConfig.getUntrackedParameter("TauToken",edm::InputTag("hpsPFTauProducer")));
      DMFindingToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("DMFindingToken",edm::InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs")));
      DMFindingOldToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("DMFindingOldToken",edm::InputTag("hpsPFTauDiscriminationByDecayModeFindingOldDMs")));
      TightIsoToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("TightIsoToken",edm::InputTag("hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits")));
      LooseIsoToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("LooseIsoToken",edm::InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits")));
      LooseAntiMuonToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("LooseAntiMuonToken",edm::InputTag("hpsPFTauDiscriminationByLooseMuonRejection3")));
      TightAntiMuonToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("TightAntiMuonToken",edm::InputTag("hpsPFTauDiscriminationByTightMuonRejection3")));
      VLooseAntiElectronToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("VLooseAntiElectronToken",edm::InputTag("hpsPFTauDiscriminationByMVA6VLooseElectronRejection")));
      LooseAntiElectronToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("LooseAntiElectronToken",edm::InputTag("hpsPFTauDiscriminationByMVA6LooseElectronRejection")));
      TightAntiElectronToken_ = consumes<reco::PFTauDiscriminator>(iConfig.getUntrackedParameter("TightAntiElectronToken",edm::InputTag("hpsPFTauDiscriminationByMVA6TightElectronRejection")));
    }    

  /*  
  caloJetToken_ = consumes<reco::CaloJetCollection>(iConfig.getUntrackedParameter("caloJetToken",edm::InputTag("ak4CaloJets")));
  //  caloJetIdToken_ = consumes<edm::ValueMap<reco::JetID> >(iConfig.getUntrackedParameter("jetIdToken",edm::InputTag("ak4JetID")));
  jetCorrectorToken_ = consumes<reco::JetCorrector>(iConfig.getUntrackedParameter<edm::InputTag>("jetCorrToken"));

  jetptThreshold_ = iConfig.getParameter<double>      ("jetptThreshold");
  maxTau_         = iConfig.getParameter<unsigned int>("maxTau");
  */

  tau           = new L1Analysis::L1AnalysisRecoTau();
  tau_data           = tau->getData();

  /*
  // set up output
  */
  tree_=fs_->make<TTree>("TauRecoTree", "TauRecoTree");
  //tree_=fs_->make<TTree>("JetRecoTree", "JetRecoTree");
  tree_->Branch("Tau",           "L1Analysis::L1AnalysisRecoTauDataFormat",         &tau_data,                32000, 3);

}


L1TauRecoTreeProducer::~L1TauRecoTreeProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void L1TauRecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  tau->Reset();
  edm::Handle<reco::PFTauCollection> recoTaus;
  iEvent.getByToken(TauToken_, recoTaus);

  edm::Handle<reco::PFTauDiscriminator> DMFindingTaus;
  iEvent.getByToken(DMFindingToken_, DMFindingTaus);

  edm::Handle<reco::PFTauDiscriminator> DMFindingOldTaus;
  iEvent.getByToken(DMFindingOldToken_, DMFindingOldTaus);

  edm::Handle<reco::PFTauDiscriminator> TightIsoTaus;
  iEvent.getByToken(TightIsoToken_, TightIsoTaus);

  edm::Handle<reco::PFTauDiscriminator> LooseIsoTaus;
  iEvent.getByToken(LooseIsoToken_, LooseIsoTaus);

  edm::Handle<reco::PFTauDiscriminator> LooseAntiMuon;
  iEvent.getByToken(LooseAntiMuonToken_, LooseAntiMuon);

  edm::Handle<reco::PFTauDiscriminator> TightAntiMuon;
  iEvent.getByToken(TightAntiMuonToken_, TightAntiMuon);

  edm::Handle<reco::PFTauDiscriminator> VLooseAntiElectron;
  iEvent.getByToken(VLooseAntiElectronToken_, VLooseAntiElectron);

  edm::Handle<reco::PFTauDiscriminator> LooseAntiElectron;
  iEvent.getByToken(LooseAntiElectronToken_, LooseAntiElectron);

  edm::Handle<reco::PFTauDiscriminator> TightAntiElectron;
  iEvent.getByToken(TightAntiElectronToken_, TightAntiElectron);

  //std::cout<<"size of recoTaus = "<<recoTaus->size()<<std::endl;

  if (recoTaus.isValid()) {
    //std::cout<<"passing here"<<std::endl;
    tau->SetTau(iEvent, iSetup, recoTaus, DMFindingOldTaus, DMFindingTaus, TightIsoTaus, LooseIsoTaus, LooseAntiMuon, TightAntiMuon, VLooseAntiElectron, LooseAntiElectron, TightAntiElectron, maxTau_);
  }
  else {
    if (!caloJetsMissing_) {edm::LogWarning("MissingProduct") << "CaloJets not found.  Branch will not be filled" << std::endl;}
    caloJetsMissing_ = true;
  }

  /*
  jet->Reset();

  // get jets  & co...
  edm::Handle<reco::CaloJetCollection> recoCaloJets;
  edm::Handle<edm::ValueMap<reco::JetID> > jetsID;
  edm::Handle<reco::JetCorrector> jetCorr;

  iEvent.getByToken(caloJetToken_, recoCaloJets);
  //iEvent.getByLabel(jetIdTag_,jetsID);
  //iEvent.getByToken(jetCorrectorToken_, jetCorr);

  if (recoCaloJets.isValid()) {
    jet->SetCaloJet(iEvent, iSetup, recoCaloJets, maxTau_); //jetsID, maxTau_);
  }
  else {
    if (!caloJetsMissing_) {edm::LogWarning("MissingProduct") << "CaloJets not found.  Branch will not be filled" << std::endl;}
    caloJetsMissing_ = true;
  }
  */

  tree_->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void
L1TauRecoTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TauRecoTreeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TauRecoTreeProducer);
