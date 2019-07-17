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
#include "TH2F.h"


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
  void checkTrackEfficiency(edm::Handle<L1TTTrackCollectionType> tkHandle);
  template <class T1, class T2> 
  void checkEfficiency(const T1 & objCollection, const T2 & tkObjCollection);
  template<class T1, class T2> 
  void checkRate(const T1 & objCollection, const T2 & tkObjCollection);
  int findGenParticle(const edm::Handle<reco::GenParticleCollection>& genH, 
			     float& pt, float&  eta, float& phi);
  int getMotherParticleId(const reco::Candidate& gp);

  int selectedL1ObjTot_;
  int selectedL1TkObjTot_;
  int selectedL1ObjEtTot_;
  int selectedL1TkObjEtTot_;


  TH1F* etaGenL1Obj;
  TH1F* etaL1Track;
  TH1F* etaL1Obj;
  TH1F* etaL1TrkObj;

  TH1F* phiGenL1Obj;
  TH1F* phiL1Track;
  TH1F* phiL1Obj;
  TH1F* phiL1TrkObj;

  TH1F* etGenL1Obj;
  TH1F* ptL1Track;
  TH1F* ptL1TrackTurnOn;
  TH1F* etL1Obj;
  TH1F* etL1ObjTurnOn;
  TH1F* etL1TrkObj;  
  TH1F* etL1TrkObjTurnOn;
  TH1F* etThrL1Obj;
  TH1F* etThrL1TrkObj;
  TH2F* etGenVsL1Obj;
  TH2F* etGenVsL1TrkObj;

  TH1F* nGenL1Obj;
  TH1F* nL1Track;
  TH1F* nL1Obj;
  TH1F* nL1TrkObj;

  std::string analysisOption_;
  std::string objectType_;
  float etaCutoff_;
  float trkPtCutoff_;
  float genPtThreshold_;
  float etThreshold_;

  int genIndex;
  float genPt;
  float genEta;
  float genPhi;

  const edm::EDGetTokenT< MuonBxCollection > muToken;
  const edm::EDGetTokenT< EGammaBxCollection > egToken;
  const edm::EDGetTokenT< std::vector< L1TTTrackType > > trackToken;
  const edm::EDGetTokenT< RegionalMuonCandBxCollection > bmtfToken;
  const edm::EDGetTokenT< RegionalMuonCandBxCollection > omtfToken;
  const edm::EDGetTokenT< RegionalMuonCandBxCollection > emtfToken;
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
  bmtfToken(consumes< RegionalMuonCandBxCollection >(iConfig.getParameter<edm::InputTag>("L1BMTFInputTag"))),
  omtfToken(consumes< RegionalMuonCandBxCollection >(iConfig.getParameter<edm::InputTag>("L1OMTFInputTag"))),
  emtfToken(consumes< RegionalMuonCandBxCollection >(iConfig.getParameter<edm::InputTag>("L1EMTFInputTag"))),
  tkPhToken(consumes< L1TkEmParticleCollection > (iConfig.getParameter<edm::InputTag>("L1TkPhotonInputTag"))),
  tkElToken(consumes< L1TkElectronParticleCollection > (iConfig.getParameter<edm::InputTag>("L1TkElectronInputTag"))),
  genToken(consumes < reco::GenParticleCollection > (iConfig.getParameter<edm::InputTag>("GenParticleInputTag")))
{

  edm::Service<TFileService> fs;
  analysisOption_ = iConfig.getParameter<std::string>("AnalysisOption");
  objectType_ = iConfig.getParameter<std::string>("ObjectTyp");
  etaCutoff_ = iConfig.getParameter<double>("EtaCutOff");
  trkPtCutoff_ = iConfig.getParameter<double>("TrackPtCutOff");
  genPtThreshold_ = iConfig.getParameter<double>("GenPtThreshold");
  etThreshold_    = iConfig.getParameter<double>("EtThreshold");

  
}
void L1TkObjectAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  
  std::ostringstream HistoName;

  HistoName.str("");
  HistoName << "NumberOfTrack";
  nL1Track    = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(),500, -0.5, 499.5);
  HistoName.str("");
  HistoName << "NumberOf" << objectType_;
  nL1Obj    = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(),500, -0.5, 499.5);
  HistoName.str("");
  HistoName << "NumberOfTrk" << objectType_;
  nL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 500, -0.5, 499.5);
  
  HistoName.str("");
  HistoName << "Eta" << objectType_;
  etaL1Obj  = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 90, -4.5, 4.5);
  HistoName.str("");
  HistoName << "EtaTrk" << objectType_;
  etaL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(),HistoName.str().c_str(), 90, -4.5, 4.5);

  HistoName.str("");
  HistoName << "Phi" << objectType_;
  phiL1Obj  = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 64, -3.2, 3.2);
  HistoName.str("");
  HistoName << "PhiTrk" << objectType_;
  phiL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(),HistoName.str().c_str(), 64, -3.2, 3.2);



  if (analysisOption_ == "Efficiency") {
    HistoName.str("");
    HistoName << "NumberOfGen" << objectType_;
    nGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 1000, -0.5, 999.5);
    HistoName.str("");
    HistoName << "EtaGen" << objectType_;
    etaGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 90, -4.5, 4.5);
    HistoName.str("");
    HistoName << "PhiGen" << objectType_;
    phiGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(),HistoName.str().c_str(), 64, -3.2, 3.2);
    HistoName.str("");
    HistoName << "EtGen" << objectType_;
    etGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);

    HistoName.str("");
    HistoName << "EtaTrack";
    etaL1Track = fs->make<TH1F>(HistoName.str().c_str(),HistoName.str().c_str(), 90, -4.5, 4.5);
    HistoName.str("");  
    HistoName << "PhiTrack";
    phiL1Track = fs->make<TH1F>(HistoName.str().c_str(),HistoName.str().c_str(), 64, -3.2, 3.2);
    HistoName.str("");
    HistoName << "PtTrack";
    ptL1Track = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
    HistoName.str("");
    HistoName << "PtTrackTurnOn";
    ptL1TrackTurnOn = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);

    HistoName.str("");
    HistoName << "EtGenVsEt" << objectType_ ;
    etGenVsL1Obj = fs->make<TH2F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5,30, -0.5, 59.5);
    HistoName.str("");
    HistoName << "EtGenVsEtTrk" << objectType_;
    etGenVsL1TrkObj = fs->make<TH2F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5, 30, -0.5, 59.5);

    HistoName.str("");
    HistoName << "Et" << objectType_;
    etL1Obj    = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
    HistoName.str("");
    HistoName << "EtTurnOn" << objectType_;
    etL1ObjTurnOn = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
    HistoName.str("");
    HistoName << "EtTrk" << objectType_;
    etL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
    HistoName.str("");
    HistoName << "EtTurnOnTrk" << objectType_;
    etL1TrkObjTurnOn = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 30, -0.5, 59.5);
  } else {

    HistoName.str("");  
    HistoName << "EtThresholdRef" << objectType_;
    etThrL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 90, 4.5, 94.5);
    HistoName.str("");
    HistoName << "EtThresholdTrk" << objectType_;
    etThrL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 90, 4.5, 94.5);
  }
  
  selectedL1ObjTot_ = 0;
  selectedL1TkObjTot_ = 0;
  selectedL1ObjEtTot_ = 0;
  selectedL1TkObjEtTot_ = 0;
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

  genIndex = -1;
  genPt   = -999.9;
  genEta  = -999.9;
  genPhi  = -999.9;
  
  // Gen Particle 
  if (analysisOption_ == "Efficiency") {
    edm::Handle<reco::GenParticleCollection> genParticleHandle;
    iEvent.getByToken(genToken, genParticleHandle);
    nGenL1Obj->Fill((*genParticleHandle.product()).size());
    genIndex = findGenParticle(genParticleHandle, genPt, genEta, genPhi);
  }
  // the L1Tracks

  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  nL1Track->Fill((*L1TTTrackHandle.product()).size());
  L1TTTrackCollectionType::const_iterator trackIter;
  if (analysisOption_ == "Efficiency" && genIndex >= 0) checkTrackEfficiency(L1TTTrackHandle);

  if (objectType_ == "Muon") { //L1TKMuon  
    /*    edm::Handle< MuonBxCollection > muonHandle;
    iEvent.getByToken(muToken, muonHandle);  
    MuonBxCollection muCollection = (*muonHandle.product());
    nL1Obj->Fill(muCollection.size(0));

    edm::Handle< L1TkMuonParticleCollection > l1TkMuonHandle;
    iEvent.getByToken(tkMuToken, l1TkMuonHandle);
    L1TkMuonParticleCollection l1TkMuCollection = (*l1TkMuonHandle.product()); 
    nL1TrkObj->Fill(l1TkMuCollection.size());

    if (analysisOption_ == "Efficiency" && genIndex >= 0) checkEfficiency(muCollection, l1TkMuCollection);
    else if (analysisOption_ == "Rate") checkRate(muCollection, l1TkMuCollection);    */
    std::cout << " Analysis with Muons are not supported at the moment " << std::endl;

  } else if (objectType_ == "Photon") {     // Level1 EGamma
    edm::Handle< EGammaBxCollection > eGammaHandle;
    iEvent.getByToken(egToken, eGammaHandle);  
    EGammaBxCollection egCollection = (*eGammaHandle.product());
    nL1Obj->Fill(egCollection.size(0));

    edm::Handle< L1TkEmParticleCollection > l1TkPhotonHandle;
    iEvent.getByToken(tkPhToken, l1TkPhotonHandle);
    L1TkEmParticleCollection l1TkPhCollection = (*l1TkPhotonHandle.product()); 
    nL1TrkObj->Fill(l1TkPhCollection.size());

    if (analysisOption_ == "Efficiency" && genIndex >= 0) checkEfficiency(egCollection, l1TkPhCollection);
    else if (analysisOption_ == "Rate") checkRate(egCollection, l1TkPhCollection);    

  } else if (objectType_ == "Electron") {     // Level1 EGamma
    edm::Handle< EGammaBxCollection > eGammaHandle;
    iEvent.getByToken(egToken, eGammaHandle);  
    EGammaBxCollection egCollection = (*eGammaHandle.product());
    nL1Obj->Fill(egCollection.size(0));

    edm::Handle< L1TkElectronParticleCollection > l1TkElectronHandle;
    iEvent.getByToken(tkElToken, l1TkElectronHandle);
    L1TkElectronParticleCollection l1TkElCollection = (*l1TkElectronHandle.product()); 
    nL1TrkObj->Fill(l1TkElCollection.size());

    if (analysisOption_ == "Efficiency" && genIndex >= 0) checkEfficiency(egCollection, l1TkElCollection);
    else if (analysisOption_ == "Rate") checkRate(egCollection, l1TkElCollection);    
  }
}
void L1TkObjectAnalyzer::endJob() {
  std::cout << " Number of Selected " << objectType_ << " : "  << selectedL1ObjTot_ << std::endl;
  std::cout << " Number of Selected Track " << objectType_ << " : "<< selectedL1TkObjTot_ << std::endl;
  std::cout << " Number of Events Proccessed  " << ievent << std::endl;
}
void L1TkObjectAnalyzer::checkTrackEfficiency(edm::Handle<L1TTTrackCollectionType> tkHandle) {
  if (genIndex < 0 ) return;

  int nL1Trk   = 0;
  int nL1TrkPt = 0;
  L1TTTrackCollectionType::const_iterator trackIter;
  for (trackIter = tkHandle->begin(); trackIter != tkHandle->end(); 
       ++trackIter) {
    float eta = trackIter->getMomentum().eta(); 
    float phi = trackIter->getMomentum().phi();
    float pt  = trackIter->getMomentum().perp(); 
    if (fabs(eta) < etaCutoff_ && pt > 0) {
      float dPhi = reco::deltaPhi(phi, genPhi);
      float dEta = (eta - genEta);
      float dR =  sqrt(dPhi*dPhi + dEta*dEta);
      if  (dR < 0.5) {
	nL1Trk++;
	if (pt > etThreshold_) nL1TrkPt++;
      }
    }
  }
  if (genPt > genPtThreshold_ && nL1Trk > 0) {
    ptL1Track->Fill(genPt);
    if (nL1TrkPt > 0 ) {
      ptL1TrackTurnOn->Fill(genPt);
      etaL1Track->Fill(genEta);
      phiL1Track->Fill(genPhi);
    }
  }
}

template<class T1, class T2> 
void L1TkObjectAnalyzer::checkEfficiency(const T1 & objCollection, const T2 & tkObjCollection) {
  if (genPt < genPtThreshold_) return;
  float dRminObj = 999.9; 
  float etObj  = -1.0;
  for (auto objIter = objCollection.begin(0);  objIter != objCollection.end(0); ++objIter) {
    float eta = objIter->hwEta(); 
    float phi = objIter->hwPhi(); 
    if ( fabs(eta) > etaCutoff_ ) continue;
    float dPhi = reco::deltaPhi( genPhi, phi);
    float dEta = (genEta - eta);
    float dR =  sqrt(dPhi*dPhi + dEta*dEta);
    if (dR < dRminObj) {
      dRminObj = dR;
      etObj  = objIter->et();
    }
  }
  if (dRminObj < 0.3) {
    selectedL1ObjTot_++;  
    etL1Obj->Fill(etObj);
    etaL1Obj->Fill(genEta);
    etGenVsL1Obj->Fill(etObj, genPt);
    if (etObj > etThreshold_)  {
      selectedL1ObjEtTot_++;  
      etL1ObjTurnOn->Fill(etObj);
      phiL1Obj->Fill(genPhi);
    }      
  }
  float dRminTkObj = 999.9; 
  float etTkObj  = -1.0;
  for (auto tkObjIter = tkObjCollection.begin(); tkObjIter != tkObjCollection.end(); ++tkObjIter) {
    if (fabs(tkObjIter->eta()) < etaCutoff_ && tkObjIter->pt() > 0) {
      //      if ( tkObjIter->getTrkPtr().isNonnull() && tkObjIter->getTrkPtr()->getMomentum().perp() <= trkPtCutoff_) continue;
      float dPhi = reco::deltaPhi(tkObjIter->phi(), genPhi);
      float dEta = (tkObjIter->eta() - genEta);
      float dR =  sqrt(dPhi*dPhi + dEta*dEta);
      if  (dR < dRminTkObj ) {
        dRminTkObj = dR;
        etTkObj = tkObjIter->et();
      }
    }
  }
  if (dRminTkObj < 0.3) {
    selectedL1TkObjTot_++;
    etL1TrkObj->Fill(etTkObj);
    etGenVsL1TrkObj->Fill(etTkObj, genPt);
    if (etTkObj > etThreshold_)  {
      selectedL1TkObjEtTot_++;
      etL1TrkObjTurnOn->Fill(etTkObj);
      etaL1TrkObj->Fill(genEta);
      phiL1TrkObj->Fill(genPhi);
    }
  }
  /*  std::cout << " Gen Info : eta, phi, Et " << genEta << " " <<  genPhi << " " << genPt << std::endl;
  std::cout << " L1Object Info : dR Gen , et " << dRminObj << " " << etObj << std::endl;
  std::cout << " L1TkObject Info : dR Gen , et " << dRminTkObj << " " << etTkObj << std::endl;
  std::cout << " Selected Candidates : L1Object, L1ObjectEt, L1TkObject, L1TkObjectEtThr " << selectedL1ObjTot_ << " " << selectedL1ObjEtTot_
	                                                                                   <<  " " << selectedL1TkObjTot_  << " " << selectedL1TkObjEtTot_ <<  std::endl;
  */
}
template<class T1, class T2> 
void L1TkObjectAnalyzer::checkRate(const T1 & objCollection, const T2 & tkObjCollection) {

  std::vector<L1Candidate> objLocal;
  objLocal.reserve(objCollection.size(0));
  for (auto it = objCollection.begin(0); it != objCollection.end(0); it++) objLocal.push_back(*it);
  sort(objLocal.begin(), objLocal.end(), L1TkAnal::EtComparator());

  int nL1Obj = 0;
  int nL1TrkObj = 0;
  
  for (auto objIter = objLocal.begin();  objIter != objLocal.end(); ++objIter) {
    float eta = objIter->eta(); 
    float phi = objIter->phi(); 
    float et  = objIter->et(); 
    if (fabs(eta) >= etaCutoff_) continue;
    nL1Obj++;
    if (nL1Obj == 1) {
      fillIntegralHistos(etThrL1Obj, et);
      if (et > 20.0) {
	etaL1Obj->Fill(eta);
	phiL1Obj->Fill(phi);
      }
    }
    float et_min; 
    float eta_min; 
    float phi_min; 
    float dRmin = 999.9;
    
    for (auto tkObjIter = tkObjCollection.begin(); tkObjIter != tkObjCollection.end(); ++tkObjIter) {
      if (fabs(tkObjIter->eta()) < etaCutoff_ && tkObjIter->et() > 0) {
	float dPhi = reco::deltaPhi(phi, tkObjIter->l1RefPhi());
	float dEta = (eta - tkObjIter->l1RefEta());

	float dR =  sqrt(dPhi*dPhi + dEta*dEta);
	if (dR < dRmin) {
	  dRmin = dR;
	  et_min = tkObjIter->et();
	  eta_min = tkObjIter->eta();
	  phi_min = tkObjIter->phi();
	}
      }
    }
    if (dRmin < 0.1) {
      nL1TrkObj++;
      if (nL1TrkObj == 1) {
	fillIntegralHistos(etThrL1TrkObj, et_min);
	if (et_min > 20.0) {
	  etaL1TrkObj->Fill(eta_min);      
	  phiL1TrkObj->Fill(phi_min);
	}      
      }
    }         
  }
  selectedL1ObjTot_ += nL1Obj;
  selectedL1TkObjTot_ += nL1TrkObj;
}
void L1TkObjectAnalyzer::fillIntegralHistos(TH1F* th, float var){
  int nbin = th->FindBin(var); 
  for (int ibin = 1; ibin < nbin+1; ibin++) th->Fill(th->GetBinCenter(ibin));
}
int L1TkObjectAnalyzer::findGenParticle(const edm::Handle<reco::GenParticleCollection>& genH, float& pt, float & eta, float& phi) {
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
  if (indx >= 0) {
    const reco::Candidate & p = (*genH)[indx];
    pt = p.pt();
    eta = p.eta();
    phi = p.phi();
    if ( fabs(eta) > etaCutoff_ || pt <= 0.0) return -1;
    etGenL1Obj->Fill(pt); 
    if (pt > genPtThreshold_) {
      etaGenL1Obj->Fill(eta); 
      phiGenL1Obj->Fill(phi); 
    }
  }
  return indx;
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
