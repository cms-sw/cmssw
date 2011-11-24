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
#include "DataFormats/Math/interface/deltaR.h"

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
  theTrackCollectionLabel = parameters.getParameter<edm::InputTag>("TrackCollection");


  ptBin_ = parameters.getParameter<int>("ptBin");
  ptMin_ = parameters.getParameter<double>("ptMin");
  ptMax_ = parameters.getParameter<double>("ptMax");

  etaBin_ = parameters.getParameter<int>("etaBin");
  etaMin_ = parameters.getParameter<double>("etaMin");
  etaMax_ = parameters.getParameter<double>("etaMax");

  phiBin_ = parameters.getParameter<int>("phiBin");
  phiMin_ = parameters.getParameter<double>("phiMin");
  phiMax_ = parameters.getParameter<double>("phiMax");



  test_TightMu_Minv  = dbe->book1D("test_TightMu_Minv"  ,"Minv",50,70,110);

  h_allProbes_pt = dbe->book1D("allProbes_pt","All Probes Pt", ptBin_, ptMin_, ptMax_);
  h_allProbes_barrel_pt = dbe->book1D("allProbes_barrel_pt","Barrel: all Probes Pt", ptBin_, ptMin_, ptMax_);
  h_allProbes_endcap_pt = dbe->book1D("allProbes_endcap_pt","Endcap: all Probes Pt", ptBin_, ptMin_, ptMax_);
  h_allProbes_eta = dbe->book1D("allProbes_eta","All Probes Eta", etaBin_, etaMin_, etaMax_);
  h_allProbes_hp_eta = dbe->book1D("allProbes_hp_eta","High Pt all Probes Eta", etaBin_, etaMin_, etaMax_);
  h_allProbes_phi = dbe->book1D("allProbes_phi","All Probes Phi", phiBin_, phiMin_, phiMax_);

  h_passProbes_TightMu_pt = dbe->book1D("passProbes_TightMu_pt","TightMu Passing Probes Pt", ptBin_ , ptMin_ , ptMax_ );
 h_passProbes_TightMu_barrel_pt = dbe->book1D("passProbes_TightMu_barrel_pt","Barrel: TightMu Passing Probes Pt", ptBin_ , ptMin_ , ptMax_ );
 h_passProbes_TightMu_endcap_pt = dbe->book1D("passProbes_TightMu_endcap_pt","Endcap: TightMu Passing Probes Pt", ptBin_ , ptMin_ , ptMax_ );
  h_passProbes_TightMu_eta = dbe->book1D("passProbes_TightMu_eta","TightMu Passing Probes #eta", etaBin_, etaMin_, etaMax_);
  h_passProbes_TightMu_hp_eta = dbe->book1D("passProbes_TightMu_hp_eta","High Pt TightMu Passing Probes #eta", etaBin_, etaMin_, etaMax_);
  h_passProbes_TightMu_phi = dbe->book1D("passProbes_TightMu_phi","TightMu Passing Probes #phi", phiBin_, phiMin_, phiMax_);



  /*h_failProbes_TightMu_pt = dbe->book1D("failProbes_TightMu_pt","TightMu Failling Probes Pt", ptBin_ , ptMin_ , ptMax_ );
  h_failProbes_TightMu_eta = dbe->book1D("failProbes_TightMu_eta","TightMu Failling Probes #eta", etaBin_, etaMin_, etaMax_);
  h_failProbes_TightMu_phi = dbe->book1D("failProbes_TightMu_phi","TightMu Failling Probes #phi", phiBin_, phiMin_, phiMax_);
  */

#ifdef DEBUG
  cout << "[EfficiencyAnalyzer] Parameters initialization DONE" <<endl;
#endif
}

void EfficiencyAnalyzer::analyze(const edm::Event & iEvent,const edm::EventSetup& iSetup) {

  LogTrace(metname)<<"[EfficiencyAnalyzer] Analyze the mu in different eta regions";
  
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(theMuonCollectionLabel, muons);


  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(theTrackCollectionLabel,tracks); /// to be read from output as "generalTracks"


  reco::BeamSpot beamSpot;
  Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel("offlineBeamSpot", beamSpotHandle);
  beamSpot = *beamSpotHandle;

  if(!muons.isValid()) return;


  // Loop on muon collection
  TLorentzVector Mu1, Mu2;
  float charge = 99.;
  float invmass = -99.;


  bool isMB = false;
  bool isME = false;

  for (reco::MuonCollection::const_iterator recoMu1 = muons->begin(); recoMu1!=muons->end(); ++recoMu1) {

    LogTrace(metname)<<"[EfficiencyAnalyzer] loop over first muons" << endl;

    //--- Define combined isolation
    reco::MuonIsolation Iso_muon = recoMu1->isolationR03();
    float combIso = (Iso_muon.emEt + Iso_muon.hadEt + Iso_muon.sumPt);  
    
    //--- Is Global Muon 
    if (!recoMu1->isGlobalMuon()) continue;

      // get the track combinig the information from both the Tracker and the Spectrometer
      reco::TrackRef recoCombinedGlbTrack1 = recoMu1->combinedMuon();    
      float muPt1 = recoCombinedGlbTrack1->pt();
      Mu1.SetPxPyPzE(recoCombinedGlbTrack1->px(), recoCombinedGlbTrack1->py(),recoCombinedGlbTrack1->pz(), recoCombinedGlbTrack1->p());
      

      //--- Define if it is a tight muon
      if (recoMu1->isGlobalMuon() && recoMu1->isTrackerMuon() && recoMu1->combinedMuon()->normalizedChi2()<10. 
	  && recoMu1->combinedMuon()->hitPattern().numberOfValidMuonHits()>0 && fabs(recoMu1->combinedMuon()->dxy(beamSpot.position()))<0.2 && recoMu1->combinedMuon()->hitPattern().numberOfValidPixelHits()>0 && recoMu1->numberOfMatches() > 1) {

	//-- is isolated muon
	if (muPt1 > 15  && (combIso/muPt1) < 0.1 ) {


	  for (reco::MuonCollection::const_iterator recoMu2 = muons->begin(); recoMu2!=muons->end(); ++recoMu2){ 
	  
	    LogTrace(metname)<<"[EfficiencyAnalyzer] loop over second muon" <<endl;
	  
	    if (recoMu2 == recoMu1) continue;

	    
	    if (recoMu2->eta() < 1.479 )  isMB = true;
	    if (recoMu2->eta() >= 1.479 ) isME = true;


	    //--> should we apply track quality cuts??? 
	    Mu2.SetPxPyPzE(recoMu2->px(), recoMu2->py(), recoMu2->pz(), recoMu2->p());

	    if (!recoMu2->isTrackerMuon()) continue;

	    if ( recoMu2->pt() < 5 ) continue;

	    if ( (recoMu1->charge())*(recoMu2->charge()) > 0 ) continue; 

	    float Minv = (Mu1+Mu2).M();
	    
	    if ( Minv < 70 ||  Minv > 110 ) continue; 

	    h_allProbes_pt->Fill(recoMu2->pt());
	    h_allProbes_eta->Fill(recoMu2->eta());
	    h_allProbes_phi->Fill(recoMu2->phi());


	    if (isMB) h_allProbes_barrel_pt->Fill(recoMu2->pt());
	    if (isME) h_allProbes_endcap_pt->Fill(recoMu2->pt());
	    if(recoMu2->pt() > 20 ) h_allProbes_hp_eta->Fill(recoMu2->eta());


	    test_TightMu_Minv->Fill(Minv);

	 
	    // Probes passing the tight muon criteria 

	    if (recoMu2->isGlobalMuon() && recoMu2->isTrackerMuon() && recoMu2->combinedMuon()->normalizedChi2()<10. && recoMu2->combinedMuon()->hitPattern().numberOfValidMuonHits()>0 && fabs(recoMu2->combinedMuon()->dxy(beamSpot.position()))<0.2 && recoMu2->combinedMuon()->hitPattern().numberOfValidPixelHits()>0 && recoMu2->numberOfMatches() > 1) { 
		 
		 h_passProbes_TightMu_pt->Fill(recoMu2->pt());
		 h_passProbes_TightMu_eta->Fill(recoMu2->eta());
		 h_passProbes_TightMu_phi->Fill(recoMu2->phi());

		 if (isMB) h_passProbes_TightMu_barrel_pt->Fill(recoMu2->pt());
		 if (isME) h_passProbes_TightMu_endcap_pt->Fill(recoMu2->pt());
		 if( recoMu2->pt() > 20 ) h_passProbes_TightMu_hp_eta->Fill(recoMu2->eta());
       
	    }
	  }
	      
	}
      }
  }
}







/*
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
*/
