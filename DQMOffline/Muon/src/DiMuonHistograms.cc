/* This Class Header */
#include "DQMOffline/Muon/src/DiMuonHistograms.h"

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

DiMuonHistograms::DiMuonHistograms(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService){ 
  parameters = pSet;
}

DiMuonHistograms::~DiMuonHistograms() { }

void DiMuonHistograms::beginJob(DQMStore * dbe) {
  
  metname = "DiMuonhistograms";
  LogTrace(metname)<<"[DiMuonHistograms] Parameters initialization";
  dbe->setCurrentFolder("Muons/DiMuonHistograms");  

  theMuonCollectionLabel = parameters.getParameter<edm::InputTag>("MuonCollection");
  
  etaBin = parameters.getParameter<int>("etaBin");
  etaBBin = parameters.getParameter<int>("etaBBin");
  etaEBin = parameters.getParameter<int>("etaEBin");
  
  etaBMin = parameters.getParameter<double>("etaBMin");
  etaBMax = parameters.getParameter<double>("etaBMax");
  etaECMin = parameters.getParameter<double>("etaECMin");
  etaECMax = parameters.getParameter<double>("etaECMax");
  
  LowMassMin = parameters.getParameter<double>("LowMassMin");  
  LowMassMax = parameters.getParameter<double>("LowMassMax");
  HighMassMin = parameters.getParameter<double>("HighMassMin");
  HighMassMax = parameters.getParameter<double>("HighMassMax");

  int nBin = 0;
  for (unsigned int iEtaRegion=0; iEtaRegion<3; iEtaRegion++){
    if (iEtaRegion==0) { EtaName = "";         nBin = etaBin;} 
    if (iEtaRegion==1) { EtaName = "_Barrel";  nBin = etaBBin;}
    if (iEtaRegion==2) { EtaName = "_EndCap";  nBin = etaEBin;}
    
    GlbGlbMuon_LM.push_back(dbe->book1D("GlbGlbMuon_LM"+EtaName,"InvMass_{GLB,GLB}"+EtaName,nBin, LowMassMin, LowMassMax));
    TrkTrkMuon_LM.push_back(dbe->book1D("TrkTrkMuon_LM"+EtaName,"InvMass_{TRK,TRK}"+EtaName,nBin, LowMassMin, LowMassMax));
    StaTrkMuon_LM.push_back(dbe->book1D("StaTrkMuon_LM"+EtaName,"InvMass_{STA,TRK}"+EtaName,nBin, LowMassMin, LowMassMax));
    
    GlbGlbMuon_HM.push_back(dbe->book1D("GlbGlbMuon_HM"+EtaName,"InvMass_{GLB,GLB}"+EtaName,nBin, HighMassMin, HighMassMax));
    TrkTrkMuon_HM.push_back(dbe->book1D("TrkTrkMuon_HM"+EtaName,"InvMass_{TRK,TRK}"+EtaName,nBin, HighMassMin, HighMassMax));
    StaTrkMuon_HM.push_back(dbe->book1D("StaTrkMuon_HM"+EtaName,"InvMass_{STA,TRK}"+EtaName,nBin, HighMassMin, HighMassMax));
    
    // arround the Z peak
    TightTightMuon.push_back(dbe->book1D("TightTightMuon"+EtaName,"InvMass_{Tight,Tight}"+EtaName,nBin, 55.0, 125.0));

    // low-mass resonances
    SoftSoftMuon.push_back(dbe->book1D("SoftSoftMuon"+EtaName,"InvMass_{Soft,Soft}"+EtaName,nBin, 5.0, 55.0));
  }
}
void DiMuonHistograms::analyze(const edm::Event & iEvent,const edm::EventSetup& iSetup) {

  LogTrace(metname)<<"[DiMuonHistograms] Analyze the mu in different eta regions";
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
  float InvMass = -99.;

  for (reco::MuonCollection::const_iterator recoMu1 = muons->begin(); recoMu1!=muons->end(); ++recoMu1) {
    LogTrace(metname)<<"[DiMuonHistograms] loop over 1st muon"<<endl;

    // Loop on second muons to fill invariant mass plots
    for (reco::MuonCollection::const_iterator recoMu2 = recoMu1; recoMu2!=muons->end(); ++recoMu2){ 
      LogTrace(metname)<<"[DiMuonHistograms] loop over 2nd muon"<<endl;
      if (recoMu1==recoMu2) continue;
      
      // Global-Global Muon
      if (recoMu1->isGlobalMuon() && recoMu2->isGlobalMuon()) {
	LogTrace(metname)<<"[DiMuonHistograms] Glb-Glb pair"<<endl;
	reco::TrackRef recoCombinedGlbTrack1 = recoMu1->combinedMuon();
	reco::TrackRef recoCombinedGlbTrack2 = recoMu2->combinedMuon();
	Mu1.SetPxPyPzE(recoCombinedGlbTrack1->px(), recoCombinedGlbTrack1->py(),recoCombinedGlbTrack1->pz(), recoCombinedGlbTrack1->p());
	Mu2.SetPxPyPzE(recoCombinedGlbTrack2->px(), recoCombinedGlbTrack2->py(),recoCombinedGlbTrack2->pz(), recoCombinedGlbTrack2->p());
	
	charge  = recoCombinedGlbTrack1->charge()*recoCombinedGlbTrack2->charge();
	if (charge < 0) {
	  InvMass = (Mu1+Mu2).M();
	  for (unsigned int iEtaRegion=0; iEtaRegion<3; iEtaRegion++){
	    if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	    if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	    if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	    
	    if(fabs(recoCombinedGlbTrack1->eta())>EtaCutMin && fabs(recoCombinedGlbTrack1->eta())<EtaCutMax && 
	       fabs(recoCombinedGlbTrack2->eta())>EtaCutMin && fabs(recoCombinedGlbTrack2->eta())<EtaCutMax){
	      if (InvMass < LowMassMax)  GlbGlbMuon_LM[iEtaRegion]->Fill(InvMass);
	      if (InvMass > HighMassMin) GlbGlbMuon_HM[iEtaRegion]->Fill(InvMass);
	    }
	  }
	}
	// Also Tight-Tight Muon Selection
	if (recoMu1->isGlobalMuon() && recoMu1->isTrackerMuon() && recoMu1->combinedMuon()->normalizedChi2()<10. 
	    && recoMu1->combinedMuon()->hitPattern().numberOfValidMuonHits()>0 && fabs(recoMu1->combinedMuon()->dxy(beamSpot.position()))<0.2 
	    && recoMu1->combinedMuon()->hitPattern().numberOfValidPixelHits()>0 && recoMu1->numberOfMatches() > 1   
	    && recoMu2->isGlobalMuon() && recoMu2->isTrackerMuon() && recoMu2->combinedMuon()->normalizedChi2()<10. 
	    && recoMu2->combinedMuon()->hitPattern().numberOfValidMuonHits()>0 && fabs(recoMu2->combinedMuon()->dxy(beamSpot.position()))<0.2 
	    && recoMu2->combinedMuon()->hitPattern().numberOfValidPixelHits()>0 && recoMu2->numberOfMatches() > 1) {
	  LogTrace(metname)<<"[DiMuonHistograms] Tight-Tight pair"<<endl;
	  for (unsigned int iEtaRegion=0; iEtaRegion<3; iEtaRegion++){
	    if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	    if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	    if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }

	    if(fabs(recoCombinedGlbTrack1->eta())>EtaCutMin && fabs(recoCombinedGlbTrack1->eta())<EtaCutMax && 
	       fabs(recoCombinedGlbTrack2->eta())>EtaCutMin && fabs(recoCombinedGlbTrack2->eta())<EtaCutMax){
	      if (InvMass > 55. && InvMass < 125.) TightTightMuon[iEtaRegion]->Fill(InvMass);
	    }
	  }
	}
      }
    
      // Now check for STA-TRK 
      if (recoMu2->isStandAloneMuon() && recoMu1->isTrackerMuon()) {
	LogTrace(metname)<<"[DiMuonHistograms] STA-Trk pair"<<endl;
	reco::TrackRef recoStaTrack = recoMu2->standAloneMuon();
	reco::TrackRef recoTrack    = recoMu1->track();
	Mu2.SetPxPyPzE(recoStaTrack->px(), recoStaTrack->py(),recoStaTrack->pz(), recoStaTrack->p());
	Mu1.SetPxPyPzE(recoTrack->px(), recoTrack->py(),recoTrack->pz(), recoTrack->p());

        charge  = recoStaTrack->charge()*recoTrack->charge();
        if (charge < 0) {
          InvMass = (Mu1+Mu2).M();
	  for (unsigned int iEtaRegion=0; iEtaRegion<3; iEtaRegion++){
	    if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	    if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	    if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	    
	    if(fabs(recoStaTrack->eta())>EtaCutMin && fabs(recoStaTrack->eta())<EtaCutMax &&
	       fabs(recoTrack->eta())>EtaCutMin && fabs(recoTrack->eta())<EtaCutMax){
	      if (InvMass < LowMassMax)  StaTrkMuon_LM[iEtaRegion]->Fill(InvMass);
	      if (InvMass > HighMassMin) StaTrkMuon_HM[iEtaRegion]->Fill(InvMass);
	    }
	  }
	}
      }
      if (recoMu1->isStandAloneMuon() && recoMu2->isTrackerMuon()) {
	LogTrace(metname)<<"[DiMuonHistograms] STA-Trk pair"<<endl;
	reco::TrackRef recoStaTrack = recoMu1->standAloneMuon();
	reco::TrackRef recoTrack    = recoMu2->track();
	Mu1.SetPxPyPzE(recoStaTrack->px(), recoStaTrack->py(),recoStaTrack->pz(), recoStaTrack->p());
	Mu2.SetPxPyPzE(recoTrack->px(), recoTrack->py(),recoTrack->pz(), recoTrack->p());

        charge  = recoStaTrack->charge()*recoTrack->charge();
        if (charge < 0) {
          InvMass = (Mu1+Mu2).M();
	  for (unsigned int iEtaRegion=0; iEtaRegion<3; iEtaRegion++){
	    if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	    if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	    if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	    
	    if(fabs(recoStaTrack->eta())>EtaCutMin && fabs(recoStaTrack->eta())<EtaCutMax &&
	       fabs(recoTrack->eta())>EtaCutMin && fabs(recoTrack->eta())<EtaCutMax){
	      if (InvMass < LowMassMax)  StaTrkMuon_LM[iEtaRegion]->Fill(InvMass);
	      if (InvMass > HighMassMin) StaTrkMuon_HM[iEtaRegion]->Fill(InvMass);
	    }
	  }
	}
      }

      // TRK-TRK dimuon 
      if (recoMu1->isTrackerMuon() && recoMu2->isTrackerMuon()) {
	LogTrace(metname)<<"[DiMuonHistograms] Trk-Trk dimuon pair"<<endl;
	reco::TrackRef recoTrack2 = recoMu2->track();
	reco::TrackRef recoTrack1 = recoMu1->track();
	Mu2.SetPxPyPzE(recoTrack2->px(), recoTrack2->py(),recoTrack2->pz(), recoTrack2->p());
	Mu1.SetPxPyPzE(recoTrack1->px(), recoTrack1->py(),recoTrack1->pz(), recoTrack1->p());
	
	charge  = recoTrack1->charge()*recoTrack2->charge();
        if (charge < 0) {
          InvMass = (Mu1+Mu2).M();
	  for (unsigned int iEtaRegion=0; iEtaRegion<3; iEtaRegion++){
	    if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	    if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	    if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	    
	    if(fabs(recoTrack1->eta())>EtaCutMin && fabs(recoTrack1->eta())<EtaCutMax &&
	       fabs(recoTrack2->eta())>EtaCutMin && fabs(recoTrack2->eta())<EtaCutMax){
	      if (InvMass < LowMassMax)  TrkTrkMuon_LM[iEtaRegion]->Fill(InvMass);
	      if (InvMass > HighMassMin) TrkTrkMuon_HM[iEtaRegion]->Fill(InvMass);
	    }
	  }
	}
	
	if (recoMu1->isTrackerMuon() && recoMu2->isTrackerMuon() &&
	    recoMu1->innerTrack()->found() > 11 && recoMu2->innerTrack()->found() &&
	    recoMu1->innerTrack()->chi2()/recoMu1->innerTrack()->ndof() < 4.0 && 
	    recoMu2->innerTrack()->chi2()/recoMu2->innerTrack()->ndof() < 4.0 &&
	    recoMu1->numberOfMatches() > 0 && recoMu2->numberOfMatches() > 0 && 
	    recoMu1->innerTrack()->hitPattern().pixelLayersWithMeasurement() > 1 &&
	    recoMu2->innerTrack()->hitPattern().pixelLayersWithMeasurement() > 1 &&
	    fabs(recoMu1->innerTrack()->dxy()) < 3.0 && fabs(recoMu1->innerTrack()->dxy()) < 3.0 &&
	    fabs(recoMu1->innerTrack()->dz()) < 15.0 && fabs(recoMu1->innerTrack()->dz()) < 15.0){
	  
	  if (charge < 0) {
	    InvMass = (Mu1+Mu2).M();
	    for (unsigned int iEtaRegion=0; iEtaRegion<3; iEtaRegion++){
	      if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4; }
	      if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax; }
	      if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax; }
	      
	      if(fabs(recoTrack1->eta())>EtaCutMin && fabs(recoTrack1->eta())<EtaCutMax &&
		 fabs(recoTrack2->eta())>EtaCutMin && fabs(recoTrack2->eta())<EtaCutMax){
		SoftSoftMuon[iEtaRegion]->Fill(InvMass); 
	      }
	    }
	  }
	} 
      }
    } //muon2
  } //Muon1
}

