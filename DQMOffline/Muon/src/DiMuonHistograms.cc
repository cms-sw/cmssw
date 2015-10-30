/* This Class Header */
#include "DQMOffline/Muon/interface/DiMuonHistograms.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TLorentzVector.h"
#include "TFile.h"
#include <vector>
#include <cmath>

/* C++ Headers */
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;
using namespace edm;

DiMuonHistograms::DiMuonHistograms(const edm::ParameterSet& pSet){
  
  // initialise parameters:
  parameters = pSet;

  
  // declare consumes:
  theMuonCollectionLabel_ = consumes<reco::MuonCollection>  (parameters.getParameter<edm::InputTag>("MuonCollection"));
  theBeamSpotLabel_       = mayConsume<reco::BeamSpot>      (parameters.getParameter<edm::InputTag>("BeamSpotLabel"));
  theVertexLabel_         = consumes<reco::VertexCollection>(parameters.getParameter<edm::InputTag>("VertexLabel"));

  etaBin = parameters.getParameter<int>("etaBin");
  etaBBin = parameters.getParameter<int>("etaBBin");
  etaEBin = parameters.getParameter<int>("etaEBin");
 
  etaBinLM = parameters.getParameter<int>("etaBinLM");
  etaBBinLM = parameters.getParameter<int>("etaBBinLM");
  etaEBinLM = parameters.getParameter<int>("etaEBinLM");
   
  etaBMin = parameters.getParameter<double>("etaBMin");
  etaBMax = parameters.getParameter<double>("etaBMax");
  etaECMin = parameters.getParameter<double>("etaECMin");
  etaECMax = parameters.getParameter<double>("etaECMax");
  
  LowMassMin = parameters.getParameter<double>("LowMassMin");  
  LowMassMax = parameters.getParameter<double>("LowMassMax");
  HighMassMin = parameters.getParameter<double>("HighMassMin");
  HighMassMax = parameters.getParameter<double>("HighMassMax");

}

DiMuonHistograms::~DiMuonHistograms() { }

void DiMuonHistograms::bookHistograms(DQMStore::IBooker & ibooker,
				      edm::Run const & /*iRun*/,
				      edm::EventSetup const & /* iSetup */){
  
  ibooker.cd();
  ibooker.setCurrentFolder("Muons/DiMuonHistograms");  

  int nBin = 0, nBinLM = 0;
  for (unsigned int iEtaRegion=0; iEtaRegion<3; iEtaRegion++){
    if (iEtaRegion==0) { EtaName = "";         nBin = etaBin;} 
    if (iEtaRegion==1) { EtaName = "_Barrel";  nBin = etaBBin;}
    if (iEtaRegion==2) { EtaName = "_EndCap";  nBin = etaEBin;}
    
    if (etaBinLM == 0) { nBinLM = nBin;} //for HeavyIons
    else{
    if (iEtaRegion==0) { EtaName = "";         nBinLM = etaBinLM;}
    if (iEtaRegion==1) { EtaName = "_Barrel";  nBinLM = etaBBinLM;}
    if (iEtaRegion==2) { EtaName = "_EndCap";  nBinLM = etaEBinLM;}
    }
    
    GlbGlbMuon_LM.push_back(ibooker.book1D("GlbGlbMuon_LM"+EtaName,"InvMass_{GLB,GLB}"+EtaName,nBinLM, LowMassMin, LowMassMax));
    TrkTrkMuon_LM.push_back(ibooker.book1D("TrkTrkMuon_LM"+EtaName,"InvMass_{TRK,TRK}"+EtaName,nBinLM, LowMassMin, LowMassMax));
    StaTrkMuon_LM.push_back(ibooker.book1D("StaTrkMuon_LM"+EtaName,"InvMass_{STA,TRK}"+EtaName,nBinLM, LowMassMin, LowMassMax));
    
    GlbGlbMuon_HM.push_back(ibooker.book1D("GlbGlbMuon_HM"+EtaName,"InvMass_{GLB,GLB}"+EtaName,nBin, HighMassMin, HighMassMax));
    TrkTrkMuon_HM.push_back(ibooker.book1D("TrkTrkMuon_HM"+EtaName,"InvMass_{TRK,TRK}"+EtaName,nBin, HighMassMin, HighMassMax));
    StaTrkMuon_HM.push_back(ibooker.book1D("StaTrkMuon_HM"+EtaName,"InvMass_{STA,TRK}"+EtaName,nBin, HighMassMin, HighMassMax));
    
    // arround the Z peak
    TightTightMuon.push_back(ibooker.book1D("TightTightMuon"+EtaName,"InvMass_{Tight,Tight}"+EtaName,nBin, 55.0, 125.0));

    // low-mass resonances
    SoftSoftMuon.push_back(ibooker.book1D("SoftSoftMuon"+EtaName,"InvMass_{Soft,Soft}"+EtaName,nBin, 5.0, 55.0));
  }
}

void DiMuonHistograms::analyze(const edm::Event & iEvent,const edm::EventSetup& iSetup){

  LogTrace(metname)<<"[DiMuonHistograms] Analyze the mu in different eta regions";
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(theMuonCollectionLabel_, muons);

  // =================================================================================
  // Look for the Primary Vertex (and use the BeamSpot instead, if you can't find it):
  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;
  unsigned int theIndexOfThePrimaryVertex = 999.;
  
  edm::Handle<reco::VertexCollection> vertex;
  iEvent.getByToken(theVertexLabel_, vertex);
  if (vertex.isValid()){
    for (unsigned int ind=0; ind<vertex->size(); ++ind) {
      if ( (*vertex)[ind].isValid() && !((*vertex)[ind].isFake()) ) {
	theIndexOfThePrimaryVertex = ind;
	break;
      }
    }
  }

  if (theIndexOfThePrimaryVertex<100) {
    posVtx = ((*vertex)[theIndexOfThePrimaryVertex]).position();
    errVtx = ((*vertex)[theIndexOfThePrimaryVertex]).error();
  }   
  else {
    LogInfo("RecoMuonValidator") << "reco::PrimaryVertex not found, use BeamSpot position instead\n";
    
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(theBeamSpotLabel_,recoBeamSpotHandle);
    reco::BeamSpot bs = *recoBeamSpotHandle;
    
    posVtx = bs.position();
    errVtx(0,0) = bs.BeamWidthX();
    errVtx(1,1) = bs.BeamWidthY();
    errVtx(2,2) = bs.sigmaZ();
  }
  
  const reco::Vertex vtx(posVtx,errVtx);
  
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

	if ( muon::isTightMuon(*recoMu1, vtx)  && 
	     muon::isTightMuon(*recoMu2, vtx) ) { 
  	  
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


	LogTrace(metname)<<"[DiMuonHistograms] Soft-Soft pair"<<endl;

	if (muon::isSoftMuon(*recoMu1, vtx)  && 
	    muon::isSoftMuon(*recoMu2, vtx) ) { 
	  
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

