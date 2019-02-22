// -*- C++ -*-
//
// Package:    L1TausAnalyzer
// Class:      L1TausAnalyzer
// 
/**\class L1TausAnalyzer L1TausAnalyzer.cc 

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

// L1Tau Objects:
#include "DataFormats/L1TrackTrigger/interface/L1TrkTauParticle.h"
#include "L1Trigger/Phase2L1Taus/interface/L1TrkTauEtComparator.h"
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

class L1TausAnalyzer : public edm::EDAnalyzer {
public:
  
  explicit L1TausAnalyzer(const edm::ParameterSet&);
  ~L1TausAnalyzer();
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  void fillIntegralHistos(TH1F* th, float var);  
  void finaliseEfficiencyHisto(TH1F* th, const int nEvtsTotal);
  template <class T1> 
  void checkEfficiency(const T1 & tkObjCollection);
  template<class T1> 
  void checkRate(const T1 & tkObjCollection);

  std::vector<unsigned int> findGenParticles(const edm::Handle<reco::GenParticleCollection>& genH, std::vector<float>& et, std::vector<float>& eta, std::vector<float>& phi );
  bool isLepton(unsigned int pdgId);

  /////////////////////////////////////////////////////
  // Histograms Definitions
  /////////////////////////////////////////////////////

  // Gen Particles 
  TH1F* etGenL1Obj;
  TH1F* etaGenL1Obj;
  TH1F* phiGenL1Obj;

  // L1-Track Objects 
  TH1F* nL1TrkObj;
  TH1F* etL1TrkObj;  
  TH1F* etaL1TrkObj;
  TH1F* phiL1TrkObj;
  TH1F* etL1TrkObjMatched;  
  TH1F* etaL1TrkObjMatched;
  TH1F* phiL1TrkObjMatched;

  // Performance 
  TH1F* etL1TrkObjTurnOn;
  TH1F* etGenObjTurnOn;
  TH1F* etThrL1TrkObj;
  TH1F* effL1TrkObj;

  // TH2
  TH2F* etGenVsL1TrkObj;

  /////////////////////////////////////////////////////
  // Variables Definitions
  /////////////////////////////////////////////////////

  // Counters 
  int ievent; 
  int selectedL1TkObjTot;
  int selectedL1TkObjEtTot;
  unsigned int nEvtsWithMaxHadTaus;

  // Booleans
  bool bFoundAllTaus;
  
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
  std::vector<float > genEts;
  std::vector<float > genEtas;
  std::vector<float > genPhis;

  // Tokens 
  const edm::EDGetTokenT< L1TrkTauParticleCollection > trktauToken;
  const edm::EDGetTokenT< L1TkEGTauParticleCollection > tkegtauToken;
  const edm::EDGetTokenT< reco::GenParticleCollection > genToken;

};

L1TausAnalyzer::L1TausAnalyzer(const edm::ParameterSet& iConfig) :
  trktauToken(consumes< L1TrkTauParticleCollection > (iConfig.getParameter<edm::InputTag>("L1TrkTauInputTag"))),
  tkegtauToken(consumes< L1TkEGTauParticleCollection > (iConfig.getParameter<edm::InputTag>("L1TkEGTauInputTag"))),
  genToken(consumes < reco::GenParticleCollection > (iConfig.getParameter<edm::InputTag>("GenParticleInputTag")))
{

  edm::Service<TFileService> fs;
  analysisOption_ = iConfig.getParameter<std::string>("AnalysisOption");
  objectType_ = iConfig.getParameter<std::string>("ObjectType");
  genEtaCutoff_ = iConfig.getParameter<double>("GenEtaCutOff");
  etaCutoff_ = iConfig.getParameter<double>("EtaCutOff");
  genPtThreshold_ = iConfig.getParameter<double>("GenPtThreshold");
  etThreshold_    = iConfig.getParameter<double>("EtThreshold");
}

void L1TausAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  
  std::ostringstream HistoName;

  // L1 Objects
  nL1TrkObj = fs->make<TH1F>("Multiplicity","Multiplicity", 50, -0.5, 49.5);
  etL1TrkObj  = fs->make<TH1F>("Et", "Et", 200, 0.5, 200.5);
  etaL1TrkObj = fs->make<TH1F>("Eta","Eta", 90, -4.5, 4.5);
  phiL1TrkObj = fs->make<TH1F>("Phi","Phi", 64, -3.2, 3.2);
  
  
  if (analysisOption_ == "Efficiency") {
    
    // Gen Particles
    etGenL1Obj  = fs->make<TH1F>("GenEt", "GenEt", 200, 0.5, 200.5);
    etaGenL1Obj = fs->make<TH1F>("GenEta", "GenEta", 90, -4.5, 4.5);
    phiGenL1Obj = fs->make<TH1F>("GenPhi","GenPhi", 64, -3.2, 3.2);

    // L1 Matched objects
    etL1TrkObjMatched  = fs->make<TH1F>("EtMatched", "EtMatched", 200, 0.5, 200.5);
    etaL1TrkObjMatched = fs->make<TH1F>("EtaMatched","EtaMatched", 90, -4.5, 4.5);
    phiL1TrkObjMatched = fs->make<TH1F>("PhiMatched","PhiMatched", 64, -3.2, 3.2);
    
    // 2D Plots
    etGenVsL1TrkObj = fs->make<TH2F>("GenEtVsEt", "GenEtVsEt", 200, 0.5, 200.5, 200, 0.5, 200.5);
    
    // Turn-on numerator plots
    etL1TrkObjTurnOn = fs->make<TH1F>("EtTurnOn", "EtTurnOn", 200, 0.5, 200.5);
    etGenObjTurnOn   = fs->make<TH1F>("GenEtTurnOn", "GenEtTurnOn", 200, 0.5, 200.5);
    
    // Efficiency plot
    effL1TrkObj = fs->make<TH1F>("EtEfficiency", "EtEfficiency", 200, 0.5, 200.5);
   
  } else {
    
    // Rate plot
    etThrL1TrkObj = fs->make<TH1F>("EtThreshold", "EtThreshold", 200, 0.5, 200.5);
  }
  
  selectedL1TkObjTot = 0;
  selectedL1TkObjEtTot = 0;
  nEvtsWithMaxHadTaus = 0;
  ievent = 0;
}

L1TausAnalyzer::~L1TausAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
L1TausAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  ievent++;  
  
  // std::cout<<"************************************************"<<std::endl; //marina

  // Clear global vectors 
  genIndices.clear();
  genEts.clear();
  genEtas.clear();
  genPhis.clear();
  bFoundAllTaus = false;

  // Gen Particles
  if (analysisOption_ == "Efficiency") {
    edm::Handle<reco::GenParticleCollection> genParticleHandle;
    iEvent.getByToken(genToken, genParticleHandle);
    genIndices = findGenParticles(genParticleHandle, genEts, genEtas, genPhis);
    
    // Check
    unsigned int nTrigTaus=0;
    for (unsigned int i=0; i < genIndices.size(); i++) {
      if (genEts.at(i) > 20.0) nTrigTaus++;  // fix-me: use config variable genPtThreshold_
    }
    if (nTrigTaus >= 2) bFoundAllTaus = true; // fix-me: use the number of taus for each sample
    if (bFoundAllTaus) nEvtsWithMaxHadTaus++;
    
  }
  //std::cout <<"found: "<<bFoundAllTaus<< "   events: "<<nEvtsWithMaxHadTaus<<std::endl;

  // TrkTau: start
  if (objectType_ == "TrkTau"){

    edm::Handle< L1TrkTauParticleCollection > l1TrkTauHandle;
    iEvent.getByToken(trktauToken, l1TrkTauHandle);
    L1TrkTauParticleCollection l1TrkTauCollection = (*l1TrkTauHandle.product()); 
    sort( l1TrkTauCollection.begin(), l1TrkTauCollection.end(), L1TrkTau::EtComparator() );

    // Plot the Properties
    nL1TrkObj->Fill(l1TrkTauCollection.size());
    for (auto tkObjIter = l1TrkTauCollection.begin(); tkObjIter != l1TrkTauCollection.end(); ++tkObjIter) {
      //if (fabs(tkObjIter->eta()) < etaCutoff_ && tkObjIter->pt() > 0) {
	etL1TrkObj  -> Fill(tkObjIter->et());
	etaL1TrkObj -> Fill(tkObjIter->eta());
	phiL1TrkObj -> Fill(tkObjIter->phi());
	//}      
    }
    
    if (analysisOption_ == "Efficiency" && genIndices.size() > 0) checkEfficiency(l1TrkTauCollection);
    else if (analysisOption_ == "Rate") checkRate(l1TrkTauCollection);    
    
  }// TrkTau: end
  

  // TkEGTau: start
  if (objectType_ == "TkEG"){
    
    edm::Handle< L1TkEGTauParticleCollection > l1TkEGTauHandle;
    iEvent.getByToken(tkegtauToken, l1TkEGTauHandle);
    L1TkEGTauParticleCollection l1TkEGTauCollection = (*l1TkEGTauHandle.product()); 
    sort( l1TkEGTauCollection.begin(), l1TkEGTauCollection.end(), L1TkEGTau::EtComparator() );

    // Plot the Properties
    nL1TrkObj->Fill(l1TkEGTauCollection.size());
    for (auto tkObjIter = l1TkEGTauCollection.begin(); tkObjIter != l1TkEGTauCollection.end(); ++tkObjIter) {
      //if (fabs(tkObjIter->eta()) < etaCutoff_ && tkObjIter->pt() > 0) {
	etL1TrkObj  -> Fill(tkObjIter->et());
	etaL1TrkObj -> Fill(tkObjIter->eta());
	phiL1TrkObj -> Fill(tkObjIter->phi());
	//}
    }
        
    if (analysisOption_ == "Efficiency" && genIndices.size() > 0) checkEfficiency(l1TkEGTauCollection);
    else if (analysisOption_ == "Rate") checkRate(l1TkEGTauCollection);    
    
  }// TkEGTau: end

  
}

void L1TausAnalyzer::endJob() {

  if (analysisOption_ == "Efficiency") {
    // Finalise efficiency histogram
    finaliseEfficiencyHisto(effL1TrkObj, nEvtsWithMaxHadTaus);
    
    // Print Efficiency Information
    std::cout << " Number of Selected " << objectType_ << " : "<< selectedL1TkObjTot << std::endl;
    std::cout << " Number of Events Proccessed  " << ievent << std::endl;
  }
  
}

template<class T1> 
void L1TausAnalyzer::checkEfficiency(const T1 & tkObjCollection) {

  std::vector<unsigned int> matchedL1TkObjIndx;

  // For-loop: All the gen objects in the event
  for (size_t i = 0; i < genIndices.size(); i++) {

    // Initializations
    float dRminTkObj = 999.9; 
    unsigned int indxTkObj = -1 ;
    float etTkObj, etaTkObj, phiTkObj;

    // Find the closest track object to the gen particle
    unsigned int iTkObj = 0;
    // For-loop: All the track objects in the event
    for (auto tkObjIter = tkObjCollection.begin(); tkObjIter != tkObjCollection.end(); ++tkObjIter) {
      iTkObj++;

      // Get seed track properties
      L1TTTrackRefPtr seedTk = tkObjIter->getSeedTrk();
      float seedPt  = seedTk->getMomentum().perp();
      float seedEta = seedTk->getMomentum().eta();
      float seedPhi = seedTk->getMomentum().phi();

      //if (fabs(seedEta) < etaCutoff_ && seedPt > 0) { //marina : fix all
	float dPhi = reco::deltaPhi(seedPhi, genPhis.at(i));
	float dEta = (seedEta - genEtas.at(i));
	float dR =  sqrt(dPhi*dPhi + dEta*dEta);
	if  (dR < dRminTkObj ) {
	  dRminTkObj = dR;
	  indxTkObj  = iTkObj;
	  etTkObj  = tkObjIter->et();
	  etaTkObj = tkObjIter->eta();
	  phiTkObj = tkObjIter->phi();
	}
	//}
    }// End-loop: All the track objects in the event

    // Apply the matching dR criteria
    if (dRminTkObj < 0.3) {
      selectedL1TkObjTot++;
      matchedL1TkObjIndx.push_back(indxTkObj);
      
      // Fill histos with properties of the matched track objects 
      etL1TrkObjMatched->Fill(etTkObj);
      etaL1TrkObjMatched->Fill(etaTkObj);
      phiL1TrkObjMatched->Fill(phiTkObj);
      
      etGenVsL1TrkObj->Fill(etTkObj, genEts.at(i)); //fix-me: change the order of the names etL1TrkObjVsGen
      
      // Fill turn-on numerator for a given Et threshold
      if (etTkObj > etThreshold_)  {
	selectedL1TkObjEtTot++;
	etL1TrkObjTurnOn->Fill(etTkObj);
	etGenObjTurnOn->Fill(genEts.at(i));
	//std::cout<<"Fill numerator-----------------"<<genEts.at(i)<<std::endl; //marina
      }
    }
    // else { std::cout<< dRminTkObj <<std::endl;} //marina
    
    // std::cout << " Gen Info : eta, phi, Et " << genEtas.at(i) << " " <<  genPhis.at(i) << " " << genEts.at(i) << std::endl;
    // std::cout << " L1TkObject Info : dR Gen , et " << dRminTkObj << " " << etTkObj << std::endl;
    // std::cout << " Selected Candidate : L1TkObject, L1TkObjectEtThr " << selectedL1TkObjTot  << " " << selectedL1TkObjEtTot <<  std::endl;
    
  } // End-loop: All gen objects in the event
  
  // Calculate efficiency
  if (matchedL1TkObjIndx.size() <= 0) return;
  if (!bFoundAllTaus) return;
  
  // Find the ET of the leading matched L1TrkObj
  float maxEt = 0;
  for (unsigned int i=0; i < matchedL1TkObjIndx.size(); i++) {
    if (tkObjCollection.at(i).et() > maxEt) maxEt = tkObjCollection.at(i).et();
  }
  // Fill and finalise efficiency histo 
  fillIntegralHistos(effL1TrkObj, maxEt);
  return;
}

template<class T1> 
void L1TausAnalyzer::checkRate(const T1 & tkObjCollection) {

  int nObj=0;
  float et; 
  
  // For-loop: All the track objects in the event
  for (auto tkObjIter = tkObjCollection.begin(); tkObjIter != tkObjCollection.end(); ++tkObjIter) {  // not needed (could just use the first object - leading)
    //if (fabs(tkObjIter->eta()) < etaCutoff_ && tkObjIter->et() > 0) { 
  
      nObj++;
      et = tkObjIter->et();
      
      // Fill rate histo for with the et of the leading track object
      if (nObj == 1) {
	fillIntegralHistos(etThrL1TrkObj, et);
      }
      //}         
  }// End-loop: All the track objects in the event

  return;
}

void L1TausAnalyzer::fillIntegralHistos(TH1F* th, float var){
  int nbin = th->FindBin(var); 
  for (int ibin = 1; ibin < nbin+1; ibin++) th->Fill(th->GetBinCenter(ibin));
}

void L1TausAnalyzer::finaliseEfficiencyHisto(TH1F* th, const int nEvtsTotal){
  
  const int nBinsX  = th->GetNbinsX()+1;
  double eff, err;
  
  // std::cout<< nBinsX << "  "<<nBinsY<<std::endl; //marina

  // For-loop: x-axis bins
  for (int i=0; i <= nBinsX; i++){
    
    const int nPass = th->GetBinContent(i);
    
    // Calculate the Efficiency
      if (nEvtsTotal == 0)
	{
	  eff = 0.0;
	  err = 0.0;
	}
      else {
	eff = double(nPass)/double(nEvtsTotal);
	err = (1.0/nEvtsTotal) * sqrt(nPass * (1.0 - nPass/nEvtsTotal) ); //Louise
      }
      //std::cout<<eff<<std::endl;
      // Update current histo bin to true eff value and error
      th->SetBinContent(i, eff);
      th->SetBinError  (i, err);
      
  }// For-loop: x-axis bins
  
  return;
  
}


std::vector<unsigned int> L1TausAnalyzer::findGenParticles(const edm::Handle<reco::GenParticleCollection>& genH, std::vector<float>& et, std::vector<float>& eta, std::vector<float>& phi ) {
  std::vector<unsigned int> indx;
  
  int pId = 0;
  
  if ((objectType_ == "TkEG") || (objectType_ == "TrkTau"))
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

    // Get the daughters of the genParticle
    const reco::GenParticleRefVector& daughters = p.daughterRefVector();

    // Determine if it's a last copy
    bool bDecaysToSelf = false;
    for (const auto& d : daughters) { 
      if (abs(d->pdgId()) == pId) {
	bDecaysToSelf = true;
	break;
      }
    }
    if (bDecaysToSelf) continue;
    
    // Determine if it's a hadronic decay
    bool bLeptonicDecay = false;
    for (const auto& d : daughters) {
      if (isLepton(d->pdgId())) {
	bLeptonicDecay = true;
	break;
      }
    }
    
    if (bLeptonicDecay) continue;

    // If it is a last copy and it is a hadronic decay keep it 
    indx.push_back(i);
    et.push_back(p.et());
    eta.push_back(p.eta());
    phi.push_back(p.phi());
    
    etGenL1Obj->Fill(p.et()); 
    etaGenL1Obj->Fill(p.eta());
    phiGenL1Obj->Fill(p.phi());

    //std::cout<<"Fill gen!! ---"<< p.et() <<std::endl; //marina
  }

  return indx;
}


bool L1TausAnalyzer::isLepton(unsigned int pdgId) {

  bool islepton = false;
  
  // Check if the genParticle is e,mu or their neutrinos
  if ((fabs(pdgId) == 11) || (fabs(pdgId) == 12) || (fabs(pdgId) == 13) || (fabs(pdgId) == 14)) {
    islepton = true;
  }

  return islepton;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TausAnalyzer);
