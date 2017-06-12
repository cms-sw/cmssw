// -*- C++ -*-
//
// Package:    L1TkObjectAnalyzer
// Class:      L1TkObjectAnalyzer
// 
/**\class L1TkObjectAnalyzer L1TkObjectAnalyzer.cc SLHCUpgradeSimulations/L1TkObjectAnalyzer/src/L1TkObjectAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"


// Gen-level stuff:
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticleFwd.h"


#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"


#include "DataFormats/Math/interface/deltaPhi.h"
#include "TH1F.h"


using namespace l1t;

namespace L1TkAnal{
  class EtComparator {
  public:
    bool operator()( const L1Candidate& a, const L1Candidate& b) const {
      double et_a = 0.0;
      double et_b = 0.0;
      if (cosh(a.eta()) > 0.0) et_a = a.energy()/cosh(a.eta());
      if (cosh(b.eta()) > 0.0) et_b = b.energy()/cosh(b.eta());
      return et_a > et_b;
    }
  };
}

//
// class declaration
//

class L1TkObjectAnalyzer : public edm::EDAnalyzer {
   public:

   typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
   typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;

  explicit L1TkObjectAnalyzer(const edm::ParameterSet&);
  ~L1TkObjectAnalyzer();
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  void fillIntegralHistos(TH1F* th, float var);  
  void scaleHistogram(TH1F* th, float fac);
  void checkMuonEfficiency(const edm::Handle< MuonBxCollection > & muHandle, 
			   const edm::Handle< L1TkMuonParticleCollection >& tkMuHandle,
			   const edm::Handle<reco::GenParticleCollection>& genH);
  void checkPhotonEfficiency(const edm::Handle< EGammaBxCollection > & egHandle, 
			   const edm::Handle< L1TkEmParticleCollection >& tkEmHandle,
			   const edm::Handle<reco::GenParticleCollection>& genH);
  void checkElectronEfficiency(const edm::Handle< EGammaBxCollection > & egHandle, 
			   const edm::Handle< L1TkElectronParticleCollection >& tkElHandle,
			   const edm::Handle<reco::GenParticleCollection>& genH);
  void checkMuonRate(const edm::Handle< MuonBxCollection > & muHandle,
		 const edm::Handle< L1TkMuonParticleCollection >& tkMuHandle);
  void checkPhotonRate(const edm::Handle< EGammaBxCollection > & egHandle,
		 const edm::Handle< L1TkEmParticleCollection >& tkEmHandle);
  void checkElectronRate(const edm::Handle< EGammaBxCollection > & egHandle,
		 const edm::Handle< L1TkElectronParticleCollection >& tkElHandle);
  int matchWithGenParticle(const edm::Handle<reco::GenParticleCollection>& genH, 
			     float& pt, float&  eta, float& phi);
  int getMotherParticleId(const reco::Candidate& gp);

  int selectedL1ObjTot_;
  int selectedL1TrkObjTot_;


  TH1F* etaGenL1Obj;
  TH1F* etaL1Obj;
  TH1F* etaL1TrkObj;

  TH1F* etaTrack_;
  TH1F* ptTrack_;

  TH1F* etGenL1Obj;
  TH1F* etL1Obj;
  TH1F* etL1ObjTurnOn;
  TH1F* etL1TrkObj;  
  TH1F* etL1TrkObjTurnOn;
  TH1F* etThrL1Obj;
  TH1F* etThrL1TrkObj;

  TH1F* nGenL1Obj;
  TH1F* nL1Obj;
  TH1F* nL1TrkObj;

  std::string analysisOption_;
  std::string objectType_;
  float etaCutoff_;
  float trkPtCutoff_;
  float genPtThreshold_;
  float etThreshold_;


  const edm::EDGetTokenT< MuonBxCollection > muToken;
  const edm::EDGetTokenT< EGammaBxCollection > egToken;
  const edm::EDGetTokenT< std::vector< L1TTTrackType > > trackToken;
  const edm::EDGetTokenT< L1TkMuonParticleCollection > tkMuToken;
  const edm::EDGetTokenT< L1TkEmParticleCollection > tkPhToken;
  const edm::EDGetTokenT< L1TkElectronParticleCollection > tkElToken;
  const edm::EDGetTokenT< reco::GenParticleCollection > genToken;

  int ievent; 
};

L1TkObjectAnalyzer::L1TkObjectAnalyzer(const edm::ParameterSet& iConfig) :
  muToken(consumes< MuonBxCollection >(iConfig.getParameter<edm::InputTag>("L1MuonInputTag"))),
  egToken(consumes< EGammaBxCollection >(iConfig.getParameter<edm::InputTag>("L1EGammaInputTag"))),
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
  tkMuToken(consumes< L1TkMuonParticleCollection > (iConfig.getParameter<edm::InputTag>("L1TkMuonInputTag"))),
  tkPhToken(consumes< L1TkEmParticleCollection > (iConfig.getParameter<edm::InputTag>("L1TkPhotonInputTag"))),
  tkElToken(consumes< L1TkElectronParticleCollection > (iConfig.getParameter<edm::InputTag>("L1TkElectronInputTag"))),
  genToken(consumes < reco::GenParticleCollection > (iConfig.getParameter<edm::InputTag>("GenParticleInputTag")))
{

  edm::Service<TFileService> fs;
  analysisOption_ = iConfig.getParameter<std::string>("AnalysisOption");
  objectType_ = iConfig.getParameter<std::string>("ObjectType");
  etaCutoff_ = iConfig.getParameter<double>("EtaCutOff");
  trkPtCutoff_ = iConfig.getParameter<double>("TrackPtCutOff");
  genPtThreshold_ = iConfig.getParameter<double>("GenPtThreshold");
  etThreshold_    = iConfig.getParameter<double>("EtThreshold");

  
}
void L1TkObjectAnalyzer::beginJob() {
  edm::Service<TFileService> fs;

  etaTrack_ = fs->make<TH1F>("Eta_Track","Eta of L1Tracks",50, -2.5, 2.5);
  ptTrack_ = fs->make<TH1F>("Pt_Track","Pt of L1Tracks",  30, -0.5, 59.5);  
  
  std::ostringstream HistoName;

  HistoName.str("");
  HistoName << "NumberOfGen" << objectType_;
  nGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 100, -0.5, 99.5);
  HistoName.str("");
  HistoName << "NumberOf" << objectType_;
  nL1Obj    = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(),100, -0.5, 99.5);
  HistoName.str("");
  HistoName << "NumberOfTrk" << objectType_;
  nL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 100, -0.5, 99.5);
  
  HistoName.str("");
  HistoName << "EtaGen" << objectType_;
  etaGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 50, -2.5, 2.5);
  HistoName.str("");
  HistoName << "Eta" << objectType_;
  etaL1Obj  = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 50, -2.5, 2.5);
  HistoName.str("");
  HistoName << "EtaTrk" << objectType_;
  etaL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(),HistoName.str().c_str(), 50, -2.5, 2.5);
  
  if (analysisOption_ == "Efficiency") {
    HistoName.str("");
    HistoName << "EtGen" << objectType_;
    etGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
    HistoName.str("");
    HistoName << "Et" << objectType_;
    etL1Obj    = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
    HistoName.str("");
    HistoName << "Et" << objectType_ << "TurnOn";
    etL1ObjTurnOn = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
    HistoName.str("");
    HistoName << "EtTrk" << objectType_;
    etL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
    HistoName.str("");
    HistoName << "EtTrk" << objectType_ << "TurnOn";
    etL1TrkObjTurnOn = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
  } else {
    HistoName.str("");
    HistoName << "EtThreshold" << objectType_<<"Ref";
    etThrL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 90, 4.5, 94.5);
    HistoName.str("");
    HistoName << "EtThresholdTrk" << objectType_;
    etThrL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 90, 4.5, 94.5);
  }
  
  selectedL1ObjTot_ = 0;
  selectedL1TrkObjTot_ = 0;
  ievent = 0;
}

L1TkObjectAnalyzer::~L1TkObjectAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
L1TkObjectAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  ievent++;  

  // the L1Tracks
  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trackIter;
  for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); 
       ++trackIter) {
    ptTrack_->Fill(trackIter->getMomentum().perp());
    etaTrack_->Fill(trackIter->getMomentum().eta());
  }
  if (objectType_ == "Muon") { //L1TKMuon  
    edm::Handle< MuonBxCollection > muonHandle;
    iEvent.getByToken(muToken, muonHandle);  
    nL1Obj->Fill((*muonHandle.product()).size(0));

    edm::Handle< L1TkMuonParticleCollection > l1TkMuonHandle;
    iEvent.getByToken(tkMuToken, l1TkMuonHandle);
    nL1TrkObj->Fill((*l1TkMuonHandle.product()).size());

    if (analysisOption_ == "Efficiency") {
      edm::Handle<reco::GenParticleCollection> genParticleHandle;
      iEvent.getByToken(genToken, genParticleHandle);
      checkMuonEfficiency(muonHandle, l1TkMuonHandle, genParticleHandle);
    } else checkMuonRate(muonHandle, l1TkMuonHandle);    
  } else if (objectType_ == "Photon") {     // Level1 EGamma
    edm::Handle< EGammaBxCollection > eGammaHandle;
    iEvent.getByToken(egToken, eGammaHandle);  
    nL1Obj->Fill((*eGammaHandle.product()).size(0));

    edm::Handle< L1TkEmParticleCollection > l1TkPhotonHandle;
    iEvent.getByToken(tkPhToken, l1TkPhotonHandle);
    nL1TrkObj->Fill((*l1TkPhotonHandle.product()).size());

    if (analysisOption_ == "Efficiency") {
      edm::Handle<reco::GenParticleCollection> genParticleHandle;
      iEvent.getByToken(genToken, genParticleHandle);
      checkPhotonEfficiency(eGammaHandle, l1TkPhotonHandle, genParticleHandle);
    } else checkPhotonRate(eGammaHandle, l1TkPhotonHandle);    
  } else if (objectType_ == "Electron") {     // Level1 EGamma
    edm::Handle< EGammaBxCollection > eGammaHandle;
    iEvent.getByToken(egToken, eGammaHandle);  
    nL1Obj->Fill((*eGammaHandle.product()).size(0));

    edm::Handle< L1TkElectronParticleCollection > l1TkElectronHandle;
    iEvent.getByToken(tkElToken, l1TkElectronHandle);
    nL1TrkObj->Fill((*l1TkElectronHandle.product()).size());

    if (analysisOption_ == "Efficiency") {
      edm::Handle<reco::GenParticleCollection> genParticleHandle;
      iEvent.getByToken(genToken, genParticleHandle);
      checkElectronEfficiency(eGammaHandle, l1TkElectronHandle, genParticleHandle);
    } else checkElectronRate(eGammaHandle, l1TkElectronHandle);    
  }
}
void L1TkObjectAnalyzer::endJob() {
  std::cout << " Number of Selected " << objectType_ << " : "  << selectedL1ObjTot_ << std::endl;
  std::cout << " Number of Selected Track " << objectType_ << " : "<< selectedL1TrkObjTot_ << std::endl;
  std::cout << " Number of Events Proccessed  " << ievent << std::endl;
}
void L1TkObjectAnalyzer::checkMuonEfficiency(const edm::Handle< MuonBxCollection > & muHandle, 
					     const edm::Handle< L1TkMuonParticleCollection >& tkMuHandle,
					     const edm::Handle<reco::GenParticleCollection>& genH) {
  MuonBxCollection muCollection = (*muHandle.product());
  L1TkMuonParticleCollection l1TkMuCollection = (*tkMuHandle.product()); 

  float genPt;
  float genEta;
  float genPhi;
  int genIndex  = matchWithGenParticle(genH, genPt, genEta, genPhi);
  if (genIndex < 0 ) return;

  int nL1Obj = 0;
  int nL1ObjEt = 0;
  MuonBxCollection::const_iterator muIter; 
  for (muIter = muCollection.begin(0);  muIter != muCollection.end(0); ++muIter) {
    float eta = muIter->eta(); 
    float phi = muIter->phi(); 
    float e   = muIter->energy();
    float et = 0;
    if (cosh(eta) > 0.0) et = e/cosh(eta);
    else et = -1.0;
    if ( fabs(eta) > etaCutoff_ || et <= 0.0) continue;
    float dPhi = reco::deltaPhi( genPhi, phi);
    float dEta = (genEta - eta);
    float dR =  sqrt(dPhi*dPhi + dEta*dEta);
    if (dR < 0.5) {
      nL1Obj++;
      if (et > etThreshold_)  nL1ObjEt++;
    }
  }
  if (genPt > genPtThreshold_) {
    if (nL1Obj > 0 ) etL1Obj->Fill(genPt);
    if (nL1ObjEt > 0) {
      etL1ObjTurnOn->Fill(genPt);
      etaL1Obj->Fill(genEta);
    }
  }

  int nL1TrkObj = 0;
  int nL1TrkObjEt = 0;
  L1TkMuonParticleCollection::const_iterator muTrkIter ;
  for (muTrkIter = l1TkMuCollection.begin(); muTrkIter != l1TkMuCollection.end(); ++muTrkIter) {
    if (fabs(muTrkIter->eta()) < etaCutoff_ && muTrkIter->pt() > 0) {
      if ( muTrkIter->getTrkPtr().isNonnull() && muTrkIter->getTrkPtr()->getMomentum().perp() <= trkPtCutoff_) continue;
      float dPhi = reco::deltaPhi(muTrkIter->phi(), genPhi);
      float dEta = (muTrkIter->eta() - genEta);
      float dR =  sqrt(dPhi*dPhi + dEta*dEta);
      if  (dR < 0.5) {
	nL1TrkObj++;
	if (muTrkIter->pt() > etThreshold_) nL1TrkObjEt++;
      }
    }
  }
  if (genPt > genPtThreshold_ && nL1TrkObj > 0) {
    etL1TrkObj->Fill(genPt);
    if (nL1TrkObjEt > 0 ) {
      etL1TrkObjTurnOn->Fill(genPt);
      etaL1TrkObj->Fill(genEta);
    }
  }

  selectedL1ObjTot_ += nL1Obj;
  selectedL1TrkObjTot_ += nL1TrkObj;
}
void L1TkObjectAnalyzer::checkPhotonEfficiency(const edm::Handle< EGammaBxCollection > & egHandle, 
				 	     const edm::Handle< L1TkEmParticleCollection >& tkPhHandle,
					     const edm::Handle<reco::GenParticleCollection>& genH) {
  EGammaBxCollection egCollection = (*egHandle.product());
  L1TkEmParticleCollection l1TkPhCollection = (*tkPhHandle.product()); 

  float genPt;
  float genEta;
  float genPhi;
  int genIndex  = matchWithGenParticle(genH, genPt, genEta, genPhi);
  if (genIndex < 0 ) return;

  int nL1Obj = 0;
  int nL1ObjEt = 0;
  EGammaBxCollection::const_iterator egIter; 
  for (egIter = egCollection.begin(0);  egIter != egCollection.end(0); ++egIter) {
    float eta = egIter->eta(); 
    float phi = egIter->phi(); 
    float e   = egIter->energy();
    float et = 0;
    if (cosh(eta) > 0.0) et = e/cosh(eta);
    else et = -1.0;
    if ( fabs(eta) > etaCutoff_ || et <= 0.0) continue;
    float dPhi = reco::deltaPhi( genPhi, phi);
    float dEta = (genEta - eta);
    float dR =  sqrt(dPhi*dPhi + dEta*dEta);
    if (dR < 0.5) {
      nL1Obj++;
      if (et > etThreshold_)  nL1ObjEt++;
    }
  }
  if (genPt > genPtThreshold_) {
    if (nL1Obj > 0 ) etL1Obj->Fill(genPt);
    if (nL1ObjEt > 0) {
      etL1ObjTurnOn->Fill(genPt);
      etaL1Obj->Fill(genEta);
    }
  }

  int nL1TrkObj = 0;
  int nL1TrkObjEt = 0;
  L1TkEmParticleCollection::const_iterator phTrkIter ;
  for (phTrkIter = l1TkPhCollection.begin(); phTrkIter != l1TkPhCollection.end(); ++phTrkIter) {
    if (fabs(phTrkIter->eta()) < etaCutoff_ && phTrkIter->pt() > 0) {
      float dPhi = reco::deltaPhi(phTrkIter->phi(), genPhi);
      float dEta = (phTrkIter->eta() - genEta);
      float dR =  sqrt(dPhi*dPhi + dEta*dEta);
      if  (dR < 0.5) {
	nL1TrkObj++;
	if (phTrkIter->pt() > etThreshold_) nL1TrkObjEt++;
      }
    }
  }
  if (genPt > genPtThreshold_ && nL1TrkObj > 0) {
    etL1TrkObj->Fill(genPt);
    if (nL1TrkObjEt > 0 ) {
      etL1TrkObjTurnOn->Fill(genPt);
      etaL1TrkObj->Fill(genEta);
    }
  }

  selectedL1ObjTot_ += nL1Obj;
  selectedL1TrkObjTot_ += nL1TrkObj;
}
void L1TkObjectAnalyzer::checkElectronEfficiency(const edm::Handle< EGammaBxCollection > & egHandle, 
					     const edm::Handle< L1TkElectronParticleCollection >& tkElHandle,
					     const edm::Handle<reco::GenParticleCollection>& genH) {
  EGammaBxCollection egCollection = (*egHandle.product());
  L1TkElectronParticleCollection l1TkElCollection = (*tkElHandle.product()); 

  float genPt;
  float genEta;
  float genPhi;
  int genIndex  = matchWithGenParticle(genH, genPt, genEta, genPhi);
  if (genIndex < 0 ) return;

  int nL1Obj = 0;
  int nL1ObjEt = 0;
  EGammaBxCollection::const_iterator egIter; 
  for (egIter = egCollection.begin(0);  egIter != egCollection.end(0); ++egIter) {
    float eta = egIter->eta(); 
    float phi = egIter->phi(); 
    float e   = egIter->energy();
    float et = 0;
    if (cosh(eta) > 0.0) et = e/cosh(eta);
    else et = -1.0;
    if ( fabs(eta) > etaCutoff_ || et <= 0.0) continue;
    float dPhi = reco::deltaPhi( genPhi, phi);
    float dEta = (genEta - eta);
    float dR =  sqrt(dPhi*dPhi + dEta*dEta);
    if (dR < 0.5) {
      nL1Obj++;
      if (et > etThreshold_)  nL1ObjEt++;
    }
  }
  if (genPt > genPtThreshold_) {
    if (nL1Obj > 0 ) etL1Obj->Fill(genPt);
    if (nL1ObjEt > 0) {
      etL1ObjTurnOn->Fill(genPt);
      etaL1Obj->Fill(genEta);
    }
  }

  int nL1TrkObj = 0;
  int nL1TrkObjEt = 0;
  L1TkElectronParticleCollection::const_iterator elTrkIter ;
  for (elTrkIter = l1TkElCollection.begin(); elTrkIter != l1TkElCollection.end(); ++elTrkIter) {
    if (fabs(elTrkIter->eta()) < etaCutoff_ && elTrkIter->pt() > 0) {
      if ( elTrkIter->getTrkPtr().isNonnull() && elTrkIter->getTrkPtr()->getMomentum().perp() <= trkPtCutoff_) continue;
      float dPhi = reco::deltaPhi(elTrkIter->phi(), genPhi);
      float dEta = (elTrkIter->eta() - genEta);
      float dR =  sqrt(dPhi*dPhi + dEta*dEta);
      if  (dR < 0.5) {
	nL1TrkObj++;
	if (elTrkIter->pt() > etThreshold_) nL1TrkObjEt++;
      }
    }
  }
  if (genPt > genPtThreshold_ && nL1TrkObj > 0) {
    etL1TrkObj->Fill(genPt);
    if (nL1TrkObjEt > 0 ) {
      etL1TrkObjTurnOn->Fill(genPt);
      etaL1TrkObj->Fill(genEta);
    }
  }

  selectedL1ObjTot_ += nL1Obj;
  selectedL1TrkObjTot_ += nL1TrkObj;
}
void L1TkObjectAnalyzer::checkMuonRate(const edm::Handle< MuonBxCollection > & muHandle, const edm::Handle< L1TkMuonParticleCollection >& tkMuHandle) {
 
  MuonBxCollection muCollection = (*muHandle.product());
  L1TkMuonParticleCollection l1TkMuCollection = (*tkMuHandle.product()); 
    
  int nL1Obj = 0;
  int nL1TrkObj = 0;
  std::vector<Muon> muonLocal;
  muonLocal.reserve(muCollection.size(0));
  MuonBxCollection::const_iterator it; 
  for (it = muCollection.begin(0); it != muCollection.end(0); it++) muonLocal.push_back(*it);
  sort(muonLocal.begin(), muonLocal.end(), L1TkAnal::EtComparator());
  std::vector<Muon>::const_iterator muIter; 
  for (muIter = muonLocal.begin();  muIter != muonLocal.end(); ++muIter) {
        
    float eta = muIter->eta(); 
    float phi = muIter->phi(); 
    float et  = muIter->et();
    if (fabs(eta) >= etaCutoff_) continue;
    nL1Obj++;
    if (nL1Obj == 1) {
      fillIntegralHistos(etThrL1Obj, et);
      if (et > 20.0) etaL1Obj->Fill(eta);      
    }
    float et_min; 
    float dRmin = 999.9;
    L1TkMuonParticleCollection::const_iterator muTrkIter ;
    for (muTrkIter = l1TkMuCollection.begin(); muTrkIter != l1TkMuCollection.end(); ++muTrkIter) {
      if ( !muTrkIter->getTrkPtr().isNonnull()) continue;

      if (fabs(muTrkIter->eta()) < etaCutoff_ && muTrkIter->et() > 0) {
	if ( muTrkIter->getTrkPtr()->getMomentum().perp() <= trkPtCutoff_) continue;
									     
	float dPhi = reco::deltaPhi(phi, muTrkIter->getMuRef()->phi());
	float dEta = (eta - muTrkIter->getMuRef()->eta());
	float dR =  sqrt(dPhi*dPhi + dEta*dEta);
	if (dR < dRmin) {
	  dRmin = dR;
	  et_min = muTrkIter->et();
	}
      }
    }
    if (dRmin < 0.1) {
      nL1TrkObj++;
      if (nL1TrkObj == 1) {
	fillIntegralHistos(etThrL1TrkObj, et_min);
	if (et_min > 20.0)  etaL1TrkObj->Fill(eta);      
      }
    }         
  }
  selectedL1ObjTot_ += nL1Obj;
  selectedL1TrkObjTot_ += nL1TrkObj;
}
void L1TkObjectAnalyzer::checkPhotonRate(const edm::Handle< EGammaBxCollection > & egHandle, const edm::Handle< L1TkEmParticleCollection >& tkPhHandle) {
 
  EGammaBxCollection egCollection = (*egHandle.product());
  L1TkEmParticleCollection l1TkPhCollection = (*tkPhHandle.product()); 
    
  int nL1Obj = 0;
  int nL1TrkObj = 0;
  std::vector<EGamma> egLocal;
  egLocal.reserve(egCollection.size(0));
  EGammaBxCollection::const_iterator it; 
  for (it = egCollection.begin(0); it != egCollection.end(0); it++) egLocal.push_back(*it);
  sort(egLocal.begin(), egLocal.end(), L1TkAnal::EtComparator());
  std::vector<EGamma>::const_iterator egIter; 
  for (egIter = egLocal.begin();  egIter != egLocal.end(); ++egIter) {
        
    float eta = egIter->eta(); 
    float phi = egIter->phi(); 
    float et  = egIter->et();
    if (fabs(eta) >= etaCutoff_) continue;
    nL1Obj++;
    if (nL1Obj == 1) {
      fillIntegralHistos(etThrL1Obj, et);
      if (et > 20.0) etaL1Obj->Fill(eta);      
    }
    float et_min; 
    float dRmin = 999.9;
    L1TkEmParticleCollection::const_iterator phTrkIter ;
    for (phTrkIter = l1TkPhCollection.begin(); phTrkIter != l1TkPhCollection.end(); ++phTrkIter) {
      if (fabs(phTrkIter->eta()) < etaCutoff_ && phTrkIter->et() > 0) {
									     
	float dPhi = reco::deltaPhi(phi, phTrkIter->getEGRef()->phi());
	float dEta = (eta - phTrkIter->getEGRef()->eta());
	float dR =  sqrt(dPhi*dPhi + dEta*dEta);
	if (dR < dRmin) {
	  dRmin = dR;
	  et_min = phTrkIter->et();
	}
      }
    }
    if (dRmin < 0.1) {
      nL1TrkObj++;
      if (nL1TrkObj == 1) {
	fillIntegralHistos(etThrL1TrkObj, et_min);
	if (et_min > 20.0)  etaL1TrkObj->Fill(eta);      
      }
    }         
  }
  selectedL1ObjTot_ += nL1Obj;
  selectedL1TrkObjTot_ += nL1TrkObj;
}
void L1TkObjectAnalyzer::checkElectronRate(const edm::Handle< EGammaBxCollection > & egHandle, const edm::Handle< L1TkElectronParticleCollection >& tkElHandle) {
 
  EGammaBxCollection egCollection = (*egHandle.product());
  L1TkElectronParticleCollection l1TkElCollection = (*tkElHandle.product()); 
    
  int nL1Obj = 0;
  int nL1TrkObj = 0;
  std::vector<EGamma> egLocal;
  egLocal.reserve(egCollection.size(0));
  EGammaBxCollection::const_iterator it; 
  for (it = egCollection.begin(0); it != egCollection.end(0); it++) egLocal.push_back(*it);
  sort(egLocal.begin(), egLocal.end(), L1TkAnal::EtComparator());
  std::vector<EGamma>::const_iterator egIter; 
  for (egIter = egLocal.begin();  egIter != egLocal.end(); ++egIter) {
        
    float eta = egIter->eta(); 
    float phi = egIter->phi(); 
    float et  = egIter->et();
    if (fabs(eta) >= etaCutoff_) continue;
    nL1Obj++;
    if (nL1Obj == 1) {
      fillIntegralHistos(etThrL1Obj, et);
      if (et > 20.0) etaL1Obj->Fill(eta);      
    }
    float et_min; 
    float dRmin = 999.9;
    L1TkElectronParticleCollection::const_iterator elTrkIter ;
    for (elTrkIter = l1TkElCollection.begin(); elTrkIter != l1TkElCollection.end(); ++elTrkIter) {
      if ( !elTrkIter->getTrkPtr().isNonnull()) continue;

      if (fabs(elTrkIter->eta()) < etaCutoff_ && elTrkIter->et() > 0) {
	if ( elTrkIter->getTrkPtr()->getMomentum().perp() <= trkPtCutoff_) continue;
									     
	float dPhi = reco::deltaPhi(phi, elTrkIter->getEGRef()->phi());
	float dEta = (eta - elTrkIter->getEGRef()->eta());
	float dR =  sqrt(dPhi*dPhi + dEta*dEta);
	if (dR < dRmin) {
	  dRmin = dR;
	  et_min = elTrkIter->et();
	}
      }
    }
    if (dRmin < 0.1) {
      nL1TrkObj++;
      if (nL1TrkObj == 1) {
	fillIntegralHistos(etThrL1TrkObj, et_min);
	if (et_min > 20.0)  etaL1TrkObj->Fill(eta);      
      }
    }         
  }
  selectedL1ObjTot_ += nL1Obj;
  selectedL1TrkObjTot_ += nL1TrkObj;
}
void L1TkObjectAnalyzer::fillIntegralHistos(TH1F* th, float var){
  int nbin = th->FindBin(var); 
  for (int ibin = 1; ibin < nbin+1; ibin++) th->Fill(th->GetBinCenter(ibin));
}
int L1TkObjectAnalyzer::matchWithGenParticle(const edm::Handle<reco::GenParticleCollection>& genH, float& pt, float & eta, float& phi) {
  int indx = -1;
  int pId = 0;
  if (objectType_ == "Muon") pId = 13;
  else if (objectType_ == "Electron") pId = 11;
  else if (objectType_ == "Photon") pId = 22;
  for (size_t i = 0; i < (*genH.product()).size(); ++i) {
    const reco::Candidate & p = (*genH)[i];
    if (abs(p.pdgId()) == pId && p.status() == 1) {
      indx=i;
      //      if (abs(getMotherParticleId(p)) == 24) indx = i;
      break;
    }
  }
  const reco::Candidate & p = (*genH)[indx];
  pt = p.pt();
  eta = p.eta();
  phi = p.phi();
  if ( fabs(eta) > etaCutoff_ || pt <= 0.0) return -1;
  if (pt > genPtThreshold_) {
    etGenL1Obj->Fill(pt); 
    etaGenL1Obj->Fill(eta); 
  }
  return indx;
}
void L1TkObjectAnalyzer::scaleHistogram(TH1F* th, float fac){
  for (Int_t i = 1; i < th->GetNbinsX()+1; ++i) {
    Double_t cont = th->GetBinContent(i);
    Double_t err = th->GetBinError(i);
    th->SetBinContent(i, cont*fac);
    th->SetBinError(i, err*fac);
  }
}
int L1TkObjectAnalyzer::getMotherParticleId(const reco::Candidate& gp) {
  int mid = -1;
  if (gp.numberOfMothers() == 0) return mid;
  const reco::Candidate* m0 = gp.mother(0);
  if (!m0)  return mid;

  mid = m0->pdgId();
  while (gp.pdgId() == mid) {

    const reco::Candidate* m = m0->mother(0);
    if (!m) {
      mid = -1;
      break;
    }   
    mid = m->pdgId();
    m0 = m;
  }
  return mid;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkObjectAnalyzer);
