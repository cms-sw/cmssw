/* This Class Header */
#include "DQMOffline/Muon/src/EfficiencyAnalyzer.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

using namespace edm;

#include "TLorentzVector.h"
#include "TFile.h"
#include <vector>
#include "math.h"


#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"


#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

/* C++ Headers */
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;
using namespace edm;

EfficiencyAnalyzer::EfficiencyAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService){ 
  parameters = pSet;
}

EfficiencyAnalyzer::~EfficiencyAnalyzer() { }

void EfficiencyAnalyzer::beginJob(DQMStore * dbe) {
#ifdef DEBUG
  cout << "[EfficiencyAnalyzer] Parameters initialization" <<endl;
#endif
  metname = "EfficiencyAnalyzer";
  LogTrace(metname)<<"[EfficiencyAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("Muons/EfficiencyAnalyzer");  
  
  theMuonCollectionLabel = parameters.getParameter<edm::InputTag>("MuonCollection");
  
  ptBin_ = parameters.getParameter<int>("ptBin");
  ptMin_ = parameters.getParameter<double>("ptMin");
  ptMax_ = parameters.getParameter<double>("ptMax");

  etaBin_ = parameters.getParameter<int>("etaBin");
  etaMin_ = parameters.getParameter<double>("etaMin");
  etaMax_ = parameters.getParameter<double>("etaMax");

  phiBin_ = parameters.getParameter<int>("phiBin");
  phiMin_ = parameters.getParameter<double>("phiMin");
  phiMax_ = parameters.getParameter<double>("phiMax");

  Eff_Numerator_pt    = dbe->book1D("Numerator_pt"   ,"Pt  Numerator for Tight Muons",ptBin_ , ptMin_ , ptMax_ );
  Eff_Numerator_eta   = dbe->book1D("Numerator_eta"  ,"Eta Numerator for Tight Muons",etaBin_, etaMin_, etaMax_);
  Eff_Numerator_phi   = dbe->book1D("Numerator_phi"  ,"Phi Numerator for Tight Muons",phiBin_, phiMin_, phiMax_);

  Eff_Denominator_pt  = dbe->book1D("Denominator_pt" ,"Pt  Denominator for Loose Muons",ptBin_ , ptMin_ , ptMax_ );
  Eff_Denominator_eta = dbe->book1D("Denominator_eta","Eta Denominator for Loose Muons",etaBin_, etaMin_, etaMax_);
  Eff_Denominator_phi = dbe->book1D("Denominator_phi","Phi Denominator for Loose Muons",phiBin_, phiMin_, phiMax_);
#ifdef DEBUG
  cout << "[EfficiencyAnalyzer] Parameters initialization DONE" <<endl;
#endif
}
void EfficiencyAnalyzer::analyze(const edm::Event & iEvent,const edm::EventSetup& iSetup) {

  LogTrace(metname)<<"[EfficiencyAnalyzer] Analyze the mu in different eta regions";
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(theMuonCollectionLabel, muons);

  reco::BeamSpot beamSpot;
  Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel("offlineBeamSpot", beamSpotHandle);
  beamSpot = *beamSpotHandle;

  if(!muons.isValid()) return;

  // Loop on muon collection
  TLorentzVector Mu1, Mu2;
  float charge = 99.;
  float invmass = -99.;

  for (reco::MuonCollection::const_iterator recoMu1 = muons->begin(); recoMu1!=muons->end(); ++recoMu1) {
    LogTrace(metname)<<"[EfficiencyAnalyzer] loop over 1st muon"<<endl;

    // Loop on second muons to fill invariant mass plots
    for (reco::MuonCollection::const_iterator recoMu2 = recoMu1; recoMu2!=muons->end(); ++recoMu2){ 
      LogTrace(metname)<<"[EfficiencyAnalyzer] loop over 2nd muon"<<endl;
      if (recoMu1==recoMu2) continue;
      
      // Global-Global Muon
      if (!recoMu1->isGlobalMuon()) continue;
      if (!recoMu2->isGlobalMuon()) continue;
      
      reco::TrackRef recoCombinedGlbTrack1 = recoMu1->combinedMuon();
      reco::TrackRef recoCombinedGlbTrack2 = recoMu2->combinedMuon();
      Mu1.SetPxPyPzE(recoCombinedGlbTrack1->px(), recoCombinedGlbTrack1->py(),recoCombinedGlbTrack1->pz(), recoCombinedGlbTrack1->p());
      Mu2.SetPxPyPzE(recoCombinedGlbTrack2->px(), recoCombinedGlbTrack2->py(),recoCombinedGlbTrack2->pz(), recoCombinedGlbTrack2->p());
      
      charge  = recoCombinedGlbTrack1->charge()*recoCombinedGlbTrack2->charge();
      invmass = (Mu1+Mu2).M();
      
      //--- Define combined isolation
      reco::MuonIsolation Iso_muon1 = recoMu1->isolationR03();
      reco::MuonIsolation Iso_muon2 = recoMu2->isolationR03();
      float combIso1 = (Iso_muon1.emEt + Iso_muon1.hadEt + Iso_muon1.sumPt);  
      float combIso2 = (Iso_muon2.emEt + Iso_muon2.hadEt + Iso_muon2.sumPt);  
      
      if (charge > 0)                      continue;
      if (invmass < 60. || invmass > 120.) continue;
      
      bool IsTightMuon1 = false;
      bool IsTightMuon2 = false;
      bool IsLooseMuon2 = false;
      bool IsLooseMuon1 = false;
      IsLooseMuon1 = (recoMu1->isGlobalMuon() && recoMu1->isTrackerMuon());
      IsTightMuon1 = (recoMu1->isGlobalMuon() && recoMu1->isTrackerMuon() && 
		      recoMu1->combinedMuon()->normalizedChi2()<10. && 
		      recoMu1->combinedMuon()->hitPattern().numberOfValidMuonHits()>0 && 
		      fabs(recoMu1->combinedMuon()->dxy(beamSpot.position()))<0.2 && 
		      recoMu1->combinedMuon()->hitPattern().numberOfValidPixelHits()>0 &&
		      recoMu1->numberOfMatches() > 1 && 
		      Mu1.Pt() > 15  && combIso1/Mu1.Pt() < 0.1); 

      IsLooseMuon2 = (recoMu2->isGlobalMuon() && recoMu2->isTrackerMuon());
      IsTightMuon2 = (recoMu2->isGlobalMuon() && recoMu2->isTrackerMuon() && 
		      recoMu2->combinedMuon()->normalizedChi2()<10. && 
		      recoMu2->combinedMuon()->hitPattern().numberOfValidMuonHits()>0 && 
		      fabs(recoMu2->combinedMuon()->dxy(beamSpot.position()))<0.2 && 
		      recoMu2->combinedMuon()->hitPattern().numberOfValidPixelHits()>0 &&
		      recoMu2->numberOfMatches() > 1 && 
		      Mu2.Pt() > 15  && combIso2/Mu2.Pt() < 0.1); 
      
      if (!IsLooseMuon1) continue;
      if (!IsLooseMuon2) continue;
      
      if (IsTightMuon1 && IsTightMuon2) {
	Eff_Numerator_pt ->Fill(Mu1.Pt());
	Eff_Numerator_eta->Fill(Mu1.Eta());
	Eff_Numerator_phi->Fill(Mu1.Phi());
      }
      Eff_Denominator_pt ->Fill(Mu1.Pt());
      Eff_Denominator_eta->Fill(Mu1.Eta());
      Eff_Denominator_phi->Fill(Mu1.Phi());
    } //muon2
  } //Muon1
}
// void EfficiencyAnalyzer::endJob(){
// #ifdef DEBUG
//   cout << "[EfficiencyAnalyzer]  endJob() "<< endl;
// #endif  
//   double ratio = 1.;
//   for (int i=1; i<ptBin_-1; i++) {
//     if (Eff_Denominator_pt->getBinContent(i) != 0) ratio = Eff_Numerator_pt->getBinContent(i) / Eff_Denominator_pt->getBinContent(i);
//     else                                           ratio = 0;
    
//     Eff_RecoMuon_pt->setBinContent(i, ratio);
//   }
  
//   for (int i=1; i<etaBin_-1; i++) {
//     if (Eff_Denominator_eta->getBinContent(i) != 0) ratio = Eff_Numerator_eta->getBinContent(i) / Eff_Denominator_eta->getBinContent(i);
//     else                                            ratio = 0;
    
//     Eff_RecoMuon_eta->setBinContent(i, ratio);
//   }
  
//   for (int i=1; i<phiBin_-1; i++) {
//     if (Eff_Denominator_phi->getBinContent(i) != 0) ratio = Eff_Numerator_phi->getBinContent(i) / Eff_Denominator_phi->getBinContent(i);
//     else                                            ratio = 0;
    
//     Eff_RecoMuon_phi->setBinContent(i, ratio);
//   }

// }
