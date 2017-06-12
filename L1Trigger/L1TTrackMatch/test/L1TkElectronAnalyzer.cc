// -*- C++ -*-
//
// Package:    L1TkElectronAnalyzer
// Class:      L1TkElectronAnalyzer
// 
/**\class L1TkElectronAnalyzer L1TkElectronAnalyzer.cc SLHCUpgradeSimulations/L1TkElectronAnalyzer/src/L1TkElectronAnalyzer.cc

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

#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticleFwd.h"


#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"


#include "DataFormats/Math/interface/deltaPhi.h"
#include "TH1F.h"


using namespace l1t;

namespace L1TkElectron{
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

class L1TkElectronAnalyzer : public edm::EDAnalyzer {
   public:

   typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
   typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;

  explicit L1TkElectronAnalyzer(const edm::ParameterSet&);
  ~L1TkElectronAnalyzer();
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  void fillIntegralHistos(TH1F* th, float var);  
  void scaleHistogram(TH1F* th, float fac);
  void checkEfficiency(const edm::Handle<reco::GenParticleCollection>& genH);
  void checkRate();
  int matchEGWithGenParticle(const edm::Handle<reco::GenParticleCollection>& genH, 
			     float& pt, float&  eta, float& phi);
  int matchEGWithSimTrack( const edm::Handle<edm::SimTrackContainer> & simH);
  int getMotherParticleId(const reco::Candidate& gp);

  int selectedEGTot_;
  int selectedEGTrkTot_;


  TH1F* etaGen_;
  TH1F* etaEGamma_;
  TH1F* etaEGammaTrk_;

  TH1F* etaTrack_;
  TH1F* ptTrack_;

  TH1F* etGen_;
  TH1F* etGenEGamma_;
  TH1F* etGenEGammaTurnOn_;
  TH1F* etGenEGammaTrk_;
  TH1F* etGenEGammaTrkTurnOn_;
  TH1F* etEGamma_;
  TH1F* etEGammaTrk_;
  TH1F* isoEGammaTrk_;

  TH1F* nGenEGamma;
  TH1F* nEGamma;
  TH1F* nEGammaTrk;

  std::string analysisOption_;
  float etaCutoff_;
  float trkPtCutoff_;
  float genPtThreshold_;
  float egEtThreshold_;

  EGammaBxCollection eGammaCollection_;
  L1TkElectronParticleCollection l1TkElectronCollection_;

  const edm::EDGetTokenT< EGammaBxCollection > egToken;
  const edm::EDGetTokenT< L1TkElectronParticleCollection > tkElToken;
  const edm::EDGetTokenT< reco::GenParticleCollection > genToken;
  const edm::EDGetTokenT< std::vector< L1TTTrackType > > trackToken;

  int ievent; 
};

L1TkElectronAnalyzer::L1TkElectronAnalyzer(const edm::ParameterSet& iConfig) :
  egToken(consumes< EGammaBxCollection >(iConfig.getParameter<edm::InputTag>("L1EGammaInputTag"))),
  tkElToken(consumes< L1TkElectronParticleCollection > (iConfig.getParameter<edm::InputTag>("L1TkElectronInputTag"))),
  genToken(consumes < reco::GenParticleCollection > (iConfig.getParameter<edm::InputTag>("GenParticleInputTag"))),
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
{

  edm::Service<TFileService> fs;
  analysisOption_ = iConfig.getParameter<std::string>("AnalysisOption");
  etaCutoff_ = iConfig.getParameter<double>("EtaCutOff");
  trkPtCutoff_ = iConfig.getParameter<double>("TrackPtCutOff");
  genPtThreshold_ = iConfig.getParameter<double>("GenPtThreshold");
  egEtThreshold_    = iConfig.getParameter<double>("EGEtThreshold");

  
}
void L1TkElectronAnalyzer::beginJob() {
  edm::Service<TFileService> fs;

  nGenEGamma = fs->make<TH1F>("nGenEGamma","No. of Generated Electrons", 100, -0.5, 99.5);
  nEGamma    = fs->make<TH1F>("nEGamma","No. of EGamma Candidates", 100, -0.5, 99.5);
  nEGammaTrk = fs->make<TH1F>("nEGammaTrk","No. of EGammaTrk Candidates", 100, -0.5, 99.5);

  etaGen_ = fs->make<TH1F>("Eta_Gen","Eta of GenParticle", 50, -2.5, 2.5);
  etaEGamma_ = fs->make<TH1F>("Eta_EGamma","Eta of EGamma", 50, -2.5, 2.5);
  etaEGammaTrk_ = fs->make<TH1F>("Eta_EGammaTrk","Eta of TrkEGamma", 50, -2.5, 2.5);
  isoEGammaTrk_ = fs->make<TH1F>("Isolation_EGammaTrk","Isolation of TrkEGamma", 200, 0.0, 2.0);
    
  etaTrack_ = fs->make<TH1F>("Eta_Track","Eta of L1Tracks",50, -2.5, 2.5);
  ptTrack_ = fs->make<TH1F>("Pt_Track","Pt of L1Tracks",  30, -0.5, 59.5);

  if (analysisOption_ == "Efficiency") {
    etGen_ = fs->make<TH1F>("GenEt","Et of GenParticle", 30, -0.5, 59.5);
    etGenEGamma_    = fs->make<TH1F>("GenEt_EGamma","Et of GenParticle (EG > 0)", 30, -0.5, 59.5);
    etGenEGammaTurnOn_ = fs->make<TH1F>("GenEt_EGammaEt","Et of GenParticle (EG > 5)", 30, -0.5, 59.5);
    etGenEGammaTrk_ = fs->make<TH1F>("GenEt_EGammaTrk","Et of GenParticle (EGTrk > 0)", 30, -0.5, 59.5);
    etGenEGammaTrkTurnOn_ = fs->make<TH1F>("GenEt_EGammaTrkEt","Et of GenParticle (EGTrk > 5)", 30, -0.5, 59.5);
  } else {
    etEGamma_ = fs->make<TH1F>("EGammaEtThresholdEvt_Ref","Et of EGamma (EventEt threshold)", 90, 4.5, 94.5);
    etEGammaTrk_ = fs->make<TH1F>("EGammaEtThresholdEvt_Track","Et of TrkEGamma( Event Et threshold)", 90, 4.5, 94.5);
  }

  selectedEGTot_ = 0;
  selectedEGTrkTot_ = 0;
  ievent = 0;
}

L1TkElectronAnalyzer::~L1TkElectronAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
L1TkElectronAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  ievent++;  

  edm::Handle< EGammaBxCollection > eGammaHandle;
  iEvent.getByToken(egToken, eGammaHandle);  
  eGammaCollection_ = (*eGammaHandle.product());
  nEGamma->Fill(eGammaCollection_.size(0));

   // the L1Tracks
   edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
   iEvent.getByToken(trackToken, L1TTTrackHandle);
   L1TTTrackCollectionType::const_iterator trackIter;

   // the L1TkElectron
  edm::Handle< L1TkElectronParticleCollection > l1tkElectronHandle;
  iEvent.getByToken(tkElToken, l1tkElectronHandle);
  l1TkElectronCollection_ = (*l1tkElectronHandle.product());
  nEGammaTrk->Fill(l1TkElectronCollection_.size());

  for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); 
       ++trackIter) {
    ptTrack_->Fill(trackIter->getMomentum().perp());
    etaTrack_->Fill(trackIter->getMomentum().eta());
  }

  if (analysisOption_ == "Efficiency") {
    edm::Handle<reco::GenParticleCollection> genParticleHandle;
    iEvent.getByToken(genToken, genParticleHandle);
    checkEfficiency(genParticleHandle);
  } else checkRate();
}
void L1TkElectronAnalyzer::endJob() {
  std::cout << " Selected EGammas " << selectedEGTot_ << std::endl;
  std::cout << " Selected Track EGammas " << selectedEGTrkTot_ << std::endl;
  std::cout << " Number of Events Proccessed  " << ievent << std::endl;
  /*  if (analysisOption_ == "Rate") {
    float scale_fac = 30000.0/ievent;
    std::cout << " Entries EG " << etEGamma_->GetEntries() << " EGTrk " << etEGammaTrk_->GetEntries() << std::endl;
    scaleHistogram(etEGamma_, scale_fac);
    scaleHistogram(etEGammaTrk_, scale_fac);
    //    etEGamma_->Scale(scale_fac);
    //    etEGammaTrk_->Scale(scale_fac);
    }*/    
}
void L1TkElectronAnalyzer::checkEfficiency(const edm::Handle<reco::GenParticleCollection>& genH) {
  float genPt;
  float genEta;
  float genPhi;
  int nSelectedEG = matchEGWithGenParticle(genH, genPt, genEta, genPhi);
  if (nSelectedEG == 0 ) return;
  int nSelectedEGTrk = 0;
  int nSelectedEGTrkEt = 0;
  std::vector<L1TkElectronParticle>::const_iterator egTrkIter ;
  for (egTrkIter = l1TkElectronCollection_.begin(); egTrkIter != l1TkElectronCollection_.end(); ++egTrkIter) {
    if (fabs(egTrkIter->eta()) < etaCutoff_ && egTrkIter->pt() > 0) {
      if ( egTrkIter->getTrkPtr().isNonnull() && egTrkIter->getTrkPtr()->getMomentum().perp() <= trkPtCutoff_) continue;
      float dPhi = reco::deltaPhi(egTrkIter->phi(), genPhi);
      float dEta = (egTrkIter->eta() - genEta);
      float dR =  sqrt(dPhi*dPhi + dEta*dEta);
      if  (dR < 0.5) {
	nSelectedEGTrk++;
	if (egTrkIter->pt() > egEtThreshold_) nSelectedEGTrkEt++;
      }
    }
  }
  if (genPt > genPtThreshold_ && nSelectedEGTrk > 0) {
    //    std::cout<< "Event # " << ievent << " Selected  EGamma "<< nSelectedEG << " matched EGamma Trk " << nSelectedEGTrk << std::endl;     
    etGenEGammaTrk_->Fill(genPt);
    if (nSelectedEGTrkEt > 0 ) {
      etGenEGammaTrkTurnOn_->Fill(genPt);
      etaEGammaTrk_->Fill(genEta);
    }
  }

  selectedEGTot_ += nSelectedEG;
  selectedEGTrkTot_ += nSelectedEGTrk;
}
void L1TkElectronAnalyzer::checkRate() {
  int nSelectedEGTrk = 0;
  int nSelectedEG = 0;
  std::vector<EGamma> eGammaLocal;
  eGammaLocal.reserve(eGammaCollection_.size(0));
  EGammaBxCollection::const_iterator it; 
  for (it = eGammaCollection_.begin(0); it != eGammaCollection_.end(0); it++) eGammaLocal.push_back(*it);
  sort(eGammaLocal.begin(), eGammaLocal.end(), L1TkElectron::EtComparator());
  std::vector<EGamma>::const_iterator egIter; 
  for (egIter = eGammaLocal.begin();  egIter != eGammaLocal.end(); ++egIter) {
        
    float eta_ele = egIter->eta(); 
    float phi_ele = egIter->phi(); 
    float et_ele = egIter->et();
    if (fabs(eta_ele) >= etaCutoff_) continue;
    nSelectedEG++;
    if (nSelectedEG == 1) {
      fillIntegralHistos(etEGamma_, et_ele);
      if (et_ele > 20.0)  etaEGamma_->Fill(eta_ele);      
    }
    float et_min; 
    float dRmin = 999.9;
    float iso_min = 999.0; 
    std::vector<L1TkElectronParticle>::const_iterator egTrkIter ;
    for (egTrkIter = l1TkElectronCollection_.begin(); egTrkIter != l1TkElectronCollection_.end(); ++egTrkIter) {
      if ( !egTrkIter->getTrkPtr().isNonnull()) continue;

      if (fabs(egTrkIter->eta()) < etaCutoff_ && egTrkIter->et() > 0) {
	if ( egTrkIter->getTrkPtr()->getMomentum().perp() <= trkPtCutoff_) continue;
									     
	float dPhi = reco::deltaPhi(phi_ele, egTrkIter->getEGRef()->phi());
	float dEta = (eta_ele - egTrkIter->getEGRef()->eta());
	float dR =  sqrt(dPhi*dPhi + dEta*dEta);
	if (dR < dRmin) {
	  dRmin = dR;
          iso_min = egTrkIter->getTrkIsol();	
	  et_min = egTrkIter->et();
	}
      }
    }
    if (dRmin < 0.1) {
      nSelectedEGTrk++;
      if (nSelectedEGTrk == 1) {
        if (et_min> egEtThreshold_) isoEGammaTrk_->Fill(iso_min);
	fillIntegralHistos(etEGammaTrk_, et_min);
	if (et_min > 20.0)  etaEGammaTrk_->Fill(eta_ele);      
      }
    }         
  }
  selectedEGTot_ += nSelectedEG;
  selectedEGTrkTot_ += nSelectedEGTrk;
}
void L1TkElectronAnalyzer::fillIntegralHistos(TH1F* th, float var){
  int nbin = th->FindBin(var); 
  for (int ibin = 1; ibin < nbin+1; ibin++) th->Fill(th->GetBinCenter(ibin));
}
int L1TkElectronAnalyzer::matchEGWithSimTrack(const edm::Handle<edm::SimTrackContainer>& simH) {
  
  edm::SimTrackContainer simTracks_ = (*simH.product());
  if ( fabs(simTracks_[0].momentum().eta())> etaCutoff_ || simTracks_[0].momentum().pt() <= 0.0) return -1;
  etGen_->Fill(simTracks_[0].momentum().pt());
  etaGen_->Fill(simTracks_[0].momentum().eta()); 
  int nEG = 0;
  int nEGEt = 0;
  EGammaBxCollection::const_iterator egIter; 
  for (egIter = eGammaCollection_.begin(0);  egIter != eGammaCollection_.end(0); ++egIter) {    
    float eta_ele = egIter->eta(); 
    //    float phi_ele = egIter->phi(); 
    float et_ele = egIter->et();
    if ( fabs(eta_ele) > etaCutoff_ || et_ele <= 0.0) continue;
    nEG++;
    if (et_ele > egEtThreshold_)   nEGEt++;
  } 

  if (nEG > 0) etGenEGamma_->Fill(simTracks_[0].momentum().pt());
  if (nEGEt > 0) etGenEGammaTurnOn_->Fill(simTracks_[0].momentum().pt());
  return nEG;
}
int L1TkElectronAnalyzer::matchEGWithGenParticle(const edm::Handle<reco::GenParticleCollection>& genH, float& pt, float & eta, float& phi) {
  int indx = 0;
  for (size_t i = 0; i < (*genH.product()).size(); ++i) {
    const reco::Candidate & p = (*genH)[i];
    if (abs(p.pdgId()) == 11 && p.status() == 1) {
      indx=1;
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
    etGen_->Fill(pt); 
    etaGen_->Fill(eta); 
  }
  int nEG = 0;
  int nEGEt = 0;
  //  float dRmin = 999.9;
  //  float eta_min = -5.0;
  EGammaBxCollection::const_iterator egIter; 
  for (egIter = eGammaCollection_.begin(0);  egIter != eGammaCollection_.end(0); ++egIter) {
    float eta_ele = egIter->eta(); 
    float phi_ele = egIter->phi(); 
    float e_ele   = egIter->energy();
    float et_ele = 0;
    if (cosh(eta_ele) > 0.0) et_ele = e_ele/cosh(eta_ele);
    else et_ele = -1.0;
    if ( fabs(eta_ele) > etaCutoff_ || et_ele <= 0.0) continue;
    float dPhi = reco::deltaPhi(phi, phi_ele);
    float dEta = (eta - eta_ele);
    float dR =  sqrt(dPhi*dPhi + dEta*dEta);
    if (dR < 0.5) {
      nEG++;
      if (et_ele > egEtThreshold_)  nEGEt++;
    }
  }
  if (pt > genPtThreshold_) {
    if (nEG > 0 && pt > genPtThreshold_) etGenEGamma_->Fill(pt);
    if (nEGEt > 0) {
      etGenEGammaTurnOn_->Fill(pt);
      etaEGamma_->Fill(eta);
    }
  }
  return nEG;
}
void L1TkElectronAnalyzer::scaleHistogram(TH1F* th, float fac){
  for (Int_t i = 1; i < th->GetNbinsX()+1; ++i) {
    Double_t cont = th->GetBinContent(i);
    Double_t err = th->GetBinError(i);
    th->SetBinContent(i, cont*fac);
    th->SetBinError(i, err*fac);
  }
}
int L1TkElectronAnalyzer::getMotherParticleId(const reco::Candidate& gp) {
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
DEFINE_FWK_MODULE(L1TkElectronAnalyzer);
