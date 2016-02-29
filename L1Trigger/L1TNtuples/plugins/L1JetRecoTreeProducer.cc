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
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

// data formats
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/JetID.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"


// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"
#include <TVector2.h>

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoJetDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMetDataFormat.h"

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

  void doPFJets(edm::Handle<reco::PFJetCollection> pfJets);
  void doPFJetCorr(edm::Handle<reco::PFJetCollection> pfJets, edm::Handle<reco::JetCorrector> pfJetCorr); 
  void doCaloJets(edm::Handle<reco::CaloJetCollection> caloJets);

  void doPFMet(edm::Handle<reco::PFMETCollection> pfMet);

  bool jetId(const reco::PFJet& jet);

public:
  L1Analysis::L1AnalysisRecoJetDataFormat*              jet_data;
  L1Analysis::L1AnalysisRecoMetDataFormat*              met_data;

private:

  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree * tree_;

  // EDM input tags
  edm::EDGetTokenT<reco::PFJetCollection>     pfJetToken_;
  //  edm::EDGetTokenT<reco::CaloJetCollection>     caloJetToken_;
  //  edm::EDGetTokenT<edm::ValueMap<reco::JetID> > caloJetIdToken_;
  edm::EDGetTokenT<reco::JetCorrector>        jecToken_;

  edm::EDGetTokenT<reco::PFMETCollection>     pfMetToken_;

  
  // debug stuff
  bool pfJetsMissing_;
  double jetptThreshold_;
  double jetetaMax_;
  unsigned int maxCl_;
  unsigned int maxJet_;
  unsigned int maxVtx_;
  unsigned int maxTrk_;

  bool pfMetMissing_;
  bool pfJetCorrMissing_;

};



L1JetRecoTreeProducer::L1JetRecoTreeProducer(const edm::ParameterSet& iConfig):
  pfJetsMissing_(false),
  pfMetMissing_(false),
  pfJetCorrMissing_(false)
{
  
  //  caloJetToken_ = consumes<reco::CaloJetCollection>(iConfig.getUntrackedParameter("caloJetToken",edm::InputTag("ak4CaloJets")));
  pfJetToken_ = consumes<reco::PFJetCollection>(iConfig.getUntrackedParameter("pfJetToken",edm::InputTag("ak4PFJetsCHS")));
  //  caloJetIdToken_ = consumes<edm::ValueMap<reco::JetID> >(iConfig.getUntrackedParameter("jetIdToken",edm::InputTag("ak4JetID")));
  jecToken_ = consumes<reco::JetCorrector>(iConfig.getUntrackedParameter<edm::InputTag>("jecToken",edm::InputTag("ak4PFCHSL1FastL2L3ResidualCorrector")));

  pfMetToken_ = consumes<reco::PFMETCollection>(iConfig.getUntrackedParameter("pfMetToken",edm::InputTag("pfMet")));

  jetptThreshold_ = iConfig.getParameter<double>      ("jetptThreshold");
  jetetaMax_       = iConfig.getParameter<double>      ("jetetaMax");
  maxJet_         = iConfig.getParameter<unsigned int>("maxJet");

  jet_data = new L1Analysis::L1AnalysisRecoJetDataFormat();
  met_data = new L1Analysis::L1AnalysisRecoMetDataFormat();

  // set up output
  tree_=fs_->make<TTree>("JetRecoTree", "JetRecoTree");
  tree_->Branch("Jet", "L1Analysis::L1AnalysisRecoJetDataFormat", &jet_data, 32000, 3);
  tree_->Branch("Sums", "L1Analysis::L1AnalysisRecoMetDataFormat", &met_data, 32000, 3);

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

  jet_data->Reset();
  met_data->Reset();
  
  // get jets
  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByToken(pfJetToken_, pfJets);

  //get sums
  edm::Handle<reco::PFMETCollection> pfMet;
  iEvent.getByToken(pfMetToken_, pfMet);
  
  // get jet ID
  //  edm::Handle<edm::ValueMap<reco::JetID> > jetsID;
  //iEvent.getByLabel(jetIdTag_,jetsID);

  edm::Handle<reco::JetCorrector> pfJetCorr;
  iEvent.getByToken(jecToken_, pfJetCorr);


  if (pfJets.isValid()) {

    jet_data->nJets=0;

    doPFJets(pfJets);

  }
  else {
    if (!pfJetsMissing_) {edm::LogWarning("MissingProduct") << "PFJets not found.  Branch will not be filled" << std::endl;}
    pfJetsMissing_ = true;
  }
 
  if (pfJetCorr.isValid()) {
 
    doPFJetCorr(pfJets,pfJetCorr);
 
  }
  else {
    if (!pfJetCorrMissing_)  {edm::LogWarning("MissingProduct") << "Jet Corrector not found.  Branch will not be filled" << std::endl;}
    pfJetCorrMissing_ = true;
  }
 
  if (pfMet.isValid()) {
 
    doPFMet(pfMet);

  }
  else {
    if (!pfMetMissing_) {edm::LogWarning("MissingProduct") << "PFMet not found.  Branch will not be filled" << std::endl;}
    pfMetMissing_ = true;
  }


  tree_->Fill();

}


void
L1JetRecoTreeProducer::doCaloJets(edm::Handle<reco::CaloJetCollection> caloJets) {


  for( auto it=caloJets->begin();
       it!=caloJets->end() && jet_data->nJets < maxJet_;
       ++it) {
    
    jet_data->et.push_back(it->et());
    jet_data->eta.push_back(it->eta());
    jet_data->phi.push_back(it->phi());
    jet_data->e.push_back(it->energy());
    jet_data->isPF.push_back(false);
    
    jet_data->eEMF.push_back(it->emEnergyFraction());
    jet_data->eEmEB.push_back(it->emEnergyInEB());
    jet_data->eEmEE.push_back(it->emEnergyInEE());
    jet_data->eEmHF.push_back(it->emEnergyInHF());
    jet_data->eHadHB.push_back(it->hadEnergyInHB());
    jet_data->eHadHE.push_back(it->hadEnergyInHE());
    jet_data->eHadHO.push_back(it->hadEnergyInHO());
    jet_data->eHadHF.push_back(it->hadEnergyInHF());
    jet_data->eMaxEcalTow.push_back(it->maxEInEmTowers());
    jet_data->eMaxHcalTow.push_back(it->maxEInHadTowers());
    jet_data->towerArea.push_back(it->towersArea());

    jet_data->nJets++;

  }


}


void
L1JetRecoTreeProducer::doPFJets(edm::Handle<reco::PFJetCollection> pfJets) {
  

  for( auto it=pfJets->begin();
       it!=pfJets->end() && jet_data->nJets < maxJet_;
       ++it) {
    jet_data->et.push_back(it->et());
    jet_data->eta.push_back(it->eta());
    jet_data->phi.push_back(it->phi());
    jet_data->e.push_back(it->energy());
    jet_data->isPF.push_back(true);

    jet_data->chef.push_back(it->chargedHadronEnergyFraction());
    jet_data->nhef.push_back(it->neutralHadronEnergyFraction());
    jet_data->pef.push_back(it->photonEnergyFraction());
    jet_data->eef.push_back(it->electronEnergyFraction());
    jet_data->mef.push_back(it->muonEnergyFraction());
    jet_data->hfhef.push_back(it->HFHadronEnergyFraction());
    jet_data->hfemef.push_back(it->HFEMEnergyFraction());
    jet_data->chMult.push_back(it->chargedHadronMultiplicity());
    jet_data->nhMult.push_back(it->neutralHadronMultiplicity());
    jet_data->phMult.push_back(it->photonMultiplicity());
    jet_data->elMult.push_back(it->electronMultiplicity());
    jet_data->muMult.push_back(it->muonMultiplicity());
    jet_data->hfhMult.push_back(it->HFHadronMultiplicity());
    jet_data->hfemMult.push_back(it->HFEMMultiplicity());

    jet_data->cemef.push_back(it->chargedEmEnergyFraction());	  
    jet_data->cmef.push_back(it->chargedMuEnergyFraction());	  
    jet_data->nemef.push_back(it->neutralEmEnergyFraction());	  
    jet_data->cMult.push_back(it->chargedMultiplicity());
    jet_data->nMult.push_back(it->neutralMultiplicity()); 
    
    jet_data->nJets++;    

  }

}


void
L1JetRecoTreeProducer::doPFJetCorr(edm::Handle<reco::PFJetCollection> pfJets, edm::Handle<reco::JetCorrector> pfJetCorr) {

  
  float corrFactor = 1.;
  uint nJets = 0;
  
  float mHx = 0;
  float mHy = 0;
  
  met_data->Ht     = 0;
  met_data->mHt    = -999.;
  met_data->mHtPhi = -999.;


  for( auto it=pfJets->begin();
       it!=pfJets->end() && nJets < maxJet_;
       ++it) {
    
    corrFactor = pfJetCorr.product()->correction(*it);
    
    jet_data->etCorr.push_back(it->et()*corrFactor);
    jet_data->corrFactor.push_back(corrFactor);

    nJets++;

    if (it->pt()*corrFactor > jetptThreshold_ && fabs(it->eta())<jetetaMax_) {
      mHx += -1.*it->px()*corrFactor;
      mHy += -1.*it->py()*corrFactor;
      met_data->Ht  += it->pt()*corrFactor;
    }

  }

  TVector2 *tv2 = new TVector2(mHx,mHy);
  met_data->mHt	   = tv2->Mod();
  met_data->mHtPhi = tv2->Phi();

  // std::vector< std::pair<float,float> > corrJetEtsAndCorrs;
  
  // //get jet correction and fill corrected jet ets and corrections
  // for( auto it=pfJets->begin(); it!=pfJets->end(); ++it) 
  //   {
  //     float corr = pfJetCorr.product()->correction(*it);
  //     std::pair<float,float> corrJetEtAndCorr(corr*it->et(),corr);
  //     corrJetEtsAndCorrs.push_back(corrJetEtAndCorr);
  //   }

  // // sort corrected jet ets and correction factors 
  // // by corrected jet et
  // std::sort(corrJetEtsAndCorrs.rbegin(),corrJetEtsAndCorrs.rend());
  
  // //fill jet data array with sorted jet ets and corr factors
  // std::vector<std::pair<float,float> >::iterator it;
  // uint nJets = 0;
    
  // for(it = corrJetEtsAndCorrs.begin(); it != corrJetEtsAndCorrs.end() && nJets < maxJet_; ++it){
  //   jet_data->etCorr.push_back(it->first);
  //   jet_data->corrFactor.push_back(it->second);
  //   nJets++
  // }

}

void
L1JetRecoTreeProducer::doPFMet(edm::Handle<reco::PFMETCollection> pfMet) {

  const reco::PFMETCollection *metCol = pfMet.product();
  const reco::PFMET theMet = metCol->front();

  met_data->met     = theMet.et();
  met_data->metPhi  = theMet.phi();
  met_data->sumEt   = theMet.sumEt();

}


bool
L1JetRecoTreeProducer::jetId(const reco::PFJet& jet) {

  bool tmp = true;

  tmp &= jet.neutralHadronEnergyFraction() < 0.9 ;
  tmp &= jet.neutralEmEnergyFraction() < 0.9 ;
  tmp &= (jet.chargedMultiplicity() + jet.neutralMultiplicity()) > 1 ;
  tmp &= jet.muonEnergyFraction() < 0.8 ;
  if (fabs(jet.eta()) < 2.4) {
    tmp &= jet.chargedHadronEnergyFraction() > 0.0 ;
    tmp &= jet.chargedMultiplicity() > 0 ;
    tmp &= jet.chargedEmEnergyFraction() < 0.9 ;
  }
  if (fabs(jet.eta()) > 3.0) {
    tmp &= jet.neutralEmEnergyFraction() < 0.9 ;
    tmp &= jet.neutralMultiplicity() > 10 ;
  }

  // our custom selection
  tmp &= jet.muonMultiplicity() == 0;
  tmp &= jet.electronMultiplicity() == 0;

  return tmp;

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
