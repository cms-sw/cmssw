// -*- C++ -*-
//
// Package:    L1TkEGTausAnalyzer
// Class:      L1TkEGTausAnalyzer
// 
/**\class L1TkEGTausAnalyzer L1TkEGTausAnalyzer.cc 

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

#include "DataFormats/L1TrackTrigger/interface/L1TkEGTauParticle.h"
#include "L1Trigger/Phase2L1Taus/interface/L1TkEGTauEtComparator.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"


#include "DataFormats/Math/interface/deltaPhi.h"
#include "TH1F.h"
#include "TH2F.h"


using namespace l1t;

//
// class declaration
//

class L1TkEGTausAnalyzer : public edm::EDAnalyzer {
public:
  
  explicit L1TkEGTausAnalyzer(const edm::ParameterSet&);
  ~L1TkEGTausAnalyzer();
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  void fillIntegralHistos(TH1F* th, float var);  
  template <class T1> 
  void checkEfficiency(const T1 & tkObjCollection);
  template<class T1> 
  void checkRate(const T1 & tkObjCollection);

  std::vector<unsigned int> findGenParticles(const edm::Handle<reco::GenParticleCollection>& genH, std::vector<float>& pt, std::vector<float>& eta, std::vector<float>& phi );

  /////////////////////////////////////////////////////
  // Histograms Definitions
  /////////////////////////////////////////////////////

  // Gen Particles 
  TH1F* etaGenL1Obj;
  TH1F* phiGenL1Obj;
  TH1F* etGenL1Obj;

  // L1-Track Objects 
  TH1F* nL1TrkObj;
  TH1F* etaL1TrkObj;
  TH1F* phiL1TrkObj;
  TH1F* etL1TrkObj;  

  // Performance 
  TH1F* etL1TrkObjTurnOn;
  TH1F* ptGenObjTurnOn;
  TH1F* etThrL1TrkObj;
  
  // TH2
  TH2F* etGenVsL1TrkObj;

  /////////////////////////////////////////////////////
  // Variables Definitions
  /////////////////////////////////////////////////////

  // Counters 
  int ievent; 
  int selectedL1TkObjTot;
  int selectedL1TkObjEtTot;
  
  // Configuration parameters
  std::string analysisOption_;
  std::string objectType_;
  float genEtaCutoff_;
  float etaCutoff_;
  float trkPtCutoff_;
  float genPtThreshold_;
  float etThreshold_;

  // Gen Particles Properties 
  std::vector<unsigned int> genIndices;
  std::vector<float > genPts;
  std::vector<float > genEtas;
  std::vector<float > genPhis;

  // Tokens 
  const edm::EDGetTokenT< std::vector< L1TTTrackType > > trackToken;
  const edm::EDGetTokenT< L1TkEGTauParticleCollection > tkegToken;
  const edm::EDGetTokenT< reco::GenParticleCollection > genToken;

};

L1TkEGTausAnalyzer::L1TkEGTausAnalyzer(const edm::ParameterSet& iConfig) :
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
  tkegToken(consumes< L1TkEGTauParticleCollection > (iConfig.getParameter<edm::InputTag>("L1TkEGInputTag"))),
  genToken(consumes < reco::GenParticleCollection > (iConfig.getParameter<edm::InputTag>("GenParticleInputTag")))
{

  edm::Service<TFileService> fs;
  analysisOption_ = iConfig.getParameter<std::string>("AnalysisOption");
  objectType_ = iConfig.getParameter<std::string>("ObjectType");
  genEtaCutoff_ = iConfig.getParameter<double>("GenEtaCutOff");
  etaCutoff_ = iConfig.getParameter<double>("EtaCutOff");
  trkPtCutoff_ = iConfig.getParameter<double>("TrackPtCutOff");
  genPtThreshold_ = iConfig.getParameter<double>("GenPtThreshold");
  etThreshold_    = iConfig.getParameter<double>("EtThreshold");
  
}

void L1TkEGTausAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  
  std::ostringstream HistoName;

  // L1-Track Objects
  HistoName.str("");
  HistoName << "NumberOf" << objectType_;
  nL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 500, -0.5, 499.5);
  
  HistoName.str("");
  HistoName << "Et" << objectType_;
  etL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 60, 0.0, 300.0);

  HistoName.str("");
  HistoName << "Eta" << objectType_;
  etaL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(),HistoName.str().c_str(), 90, -4.5, 4.5);

  HistoName.str("");
  HistoName << "Phi" << objectType_;
  phiL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(),HistoName.str().c_str(), 64, -3.2, 3.2);

  if (analysisOption_ == "Efficiency") {

    // Gen Particles
    HistoName.str("");
    HistoName << "EtaGen" << objectType_;
    etaGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 90, -4.5, 4.5);
    HistoName.str("");
    HistoName << "PhiGen" << objectType_;
    phiGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(),HistoName.str().c_str(), 64, -3.2, 3.2);
    HistoName.str("");
    HistoName << "EtGen" << objectType_;
    etGenL1Obj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 60, 0.0, 300.0);

    // 2D Plots
    HistoName.str("");
    HistoName << "EtGenVsEt" << objectType_;
    etGenVsL1TrkObj = fs->make<TH2F>(HistoName.str().c_str(), HistoName.str().c_str(), 60, 0.0, 300.0, 60, 0.0, 300.0);
    
    // Turn-on numerator plots
    HistoName.str("");
    HistoName << "EtTurnOn" << objectType_;
    etL1TrkObjTurnOn = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 60, 0.0, 300.0);
    HistoName.str("");
    HistoName << "PtTurnOn" << objectType_;
    ptGenObjTurnOn = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 60, 0.0, 300.0);
    
  } else {
    
    // Rate plot
    HistoName.str("");
    HistoName << "EtThreshold" << objectType_;
    etThrL1TrkObj = fs->make<TH1F>(HistoName.str().c_str(), HistoName.str().c_str(), 90, 4.5, 94.5);
  }
  
  selectedL1TkObjTot = 0;
  selectedL1TkObjEtTot = 0;
  ievent = 0;
}

L1TkEGTausAnalyzer::~L1TkEGTausAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
L1TkEGTausAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  ievent++;  
  
  // Clear global vectors 
  genIndices.clear();
  genPts.clear();
  genEtas.clear();
  genPhis.clear();

  // Gen Particles
  if (analysisOption_ == "Efficiency") {
    edm::Handle<reco::GenParticleCollection> genParticleHandle;
    iEvent.getByToken(genToken, genParticleHandle);
    genIndices = findGenParticles(genParticleHandle, genPts, genEtas, genPhis);
  }

  // TkEG: start
  if (objectType_ == "TkEG"){

    edm::Handle< L1TkEGTauParticleCollection > l1TkEGHandle;
    iEvent.getByToken(tkegToken, l1TkEGHandle);
    L1TkEGTauParticleCollection l1TkEGCollection = (*l1TkEGHandle.product()); 
    nL1TrkObj->Fill(l1TkEGCollection.size());
    
    sort( l1TkEGCollection.begin(), l1TkEGCollection.end(), L1TkEG::EtComparator() );

    if (analysisOption_ == "Efficiency" && genIndices.size() > 0) checkEfficiency(l1TkEGCollection);
    else if (analysisOption_ == "Rate") checkRate(l1TkEGCollection);    
    
  }// TkEG: end

  
}

void L1TkEGTausAnalyzer::endJob() {
  std::cout << " Number of Selected " << objectType_ << " : "<< selectedL1TkObjTot << std::endl;
  std::cout << " Number of Events Proccessed  " << ievent << std::endl;
}


template<class T1> 
void L1TkEGTausAnalyzer::checkEfficiency(const T1 & tkObjCollection) {

  // For-loop: All the gen objects in the event
  for (size_t i = 0; i < genIndices.size(); i++) {

    // Initializations
    float dRminTkObj = 999.9; 
    float etTkObj, etaTkObj, phiTkObj;

    // Find the closest track object to the gen particle
    // For-loop: All the track objects in the event
    for (auto tkObjIter = tkObjCollection.begin(); tkObjIter != tkObjCollection.end(); ++tkObjIter) {
      if (fabs(tkObjIter->eta()) < etaCutoff_ && tkObjIter->pt() > 0) {
	float dPhi = reco::deltaPhi(tkObjIter->phi(), genPhis.at(i));
	float dEta = (tkObjIter->eta() - genEtas.at(i));
	float dR =  sqrt(dPhi*dPhi + dEta*dEta);
	if  (dR < dRminTkObj ) {
	  dRminTkObj = dR;
	  etTkObj  = tkObjIter->et();
	  etaTkObj = tkObjIter->eta();
	  phiTkObj = tkObjIter->phi();
	}
      }
    }// End-loop: All the track objects in the event
    
    // Apply the matching dR criteria
    if (dRminTkObj < 0.3) {
      selectedL1TkObjTot++;

      // Fill histos with properties of the matched track objects 
      etL1TrkObj->Fill(etTkObj);
      etaL1TrkObj->Fill(etaTkObj);
      phiL1TrkObj->Fill(phiTkObj);

      etGenVsL1TrkObj->Fill(etTkObj, genPts.at(i));

      // Fill turn-on numerator for a given Et threshold
      if (etTkObj > etThreshold_)  {
	selectedL1TkObjEtTot++;
	etL1TrkObjTurnOn->Fill(etTkObj);
	ptGenObjTurnOn->Fill(genPts.at(i));
      }
    }

  // std::cout << " Gen Info : eta, phi, Et " << genEtas.at(i) << " " <<  genPhis.at(i) << " " << genPts.at(i) << std::endl;
  // std::cout << " L1TkObject Info : dR Gen , et " << dRminTkObj << " " << etTkObj << std::endl;
  // std::cout << " Selected Candidate : L1TkObject, L1TkObjectEtThr " << selectedL1TkObjTot  << " " << selectedL1TkObjEtTot <<  std::endl;

  } // End-loop: All gen objects in the event

  
  return;
}

template<class T1> 
void L1TkEGTausAnalyzer::checkRate(const T1 & tkObjCollection) {

  int nObj=0;
  float et; 
  
  // For-loop: All the track objects in the event
  for (auto tkObjIter = tkObjCollection.begin(); tkObjIter != tkObjCollection.end(); ++tkObjIter) {  // not needed (could just use the first object - leading)
    if (fabs(tkObjIter->eta()) < etaCutoff_ && tkObjIter->et() > 0) { 
  
      nObj++;
      et = tkObjIter->et();
      
      // Fill rate histo for with the et of the leading track object
      if (nObj == 1) {
	fillIntegralHistos(etThrL1TrkObj, et);
      }
    }         
  }// End-loop: All the track objects in the event

  return;
}

void L1TkEGTausAnalyzer::fillIntegralHistos(TH1F* th, float var){
  int nbin = th->FindBin(var); 
  for (int ibin = 1; ibin < nbin+1; ibin++) th->Fill(th->GetBinCenter(ibin));
}


std::vector<unsigned int> L1TkEGTausAnalyzer::findGenParticles(const edm::Handle<reco::GenParticleCollection>& genH, std::vector<float>& pt, std::vector<float>& eta, std::vector<float>& phi ) {
  std::vector<unsigned int> indx;

  int pId = 0;

  if (objectType_ == "TkEG") 
    {
      pId = 15;
    }
  
  const reco::GenParticleCollection& genParticles = *genH;
  unsigned int i=0;
  for(const auto& p : genParticles) {
    i++;

    if (fabs(p.eta()) > genEtaCutoff_ || p.pt() <= 0.0) continue;
    if (p.pt() < genPtThreshold_) continue;
    if (abs(p.pdgId()) != pId) continue;

    // Determine if it's a last copy
    bool bDecaysToSelf = false;

    const reco::GenParticleRefVector& daughters = p.daughterRefVector();

    for (const auto& d : daughters )
      {
       
	if (abs(d->pdgId()) == pId)
	  {
	    bDecaysToSelf = true;
	    break;
	  }
      }
    
    if (bDecaysToSelf) continue;
    
    // If it is a last copy keep it 
    indx.push_back(i);
    pt.push_back(p.pt());
    eta.push_back(p.eta());
    phi.push_back(p.phi());
    
    etGenL1Obj->Fill(p.et()); 
    etaGenL1Obj->Fill(p.eta());
    phiGenL1Obj->Fill(p.phi());
  }

  return indx;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkEGTausAnalyzer);
