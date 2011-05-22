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


using namespace edm;

#include "TLorentzVector.h"
#include "TFile.h"
#include <vector>
#include "math.h"


#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"


#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

//#define DEBUG
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

//#ifdef DEBUG   
//  cout<<"Get Muon Collection"<<endl;
//#endif 
  theMuonCollectionLabel = parameters.getParameter<edm::InputTag>("MuonCollection");
  
  etaBin = parameters.getParameter<int>("etaBin");
  etaBBin = parameters.getParameter<int>("etaBBin");
  etaEBin = parameters.getParameter<int>("etaEBin");
  etaOvlpBin = parameters.getParameter<int>("etaOvlpBin");
  
  etaBMin = parameters.getParameter<double>("etaBMin");
  etaBMax = parameters.getParameter<double>("etaBMax");
  etaECMin = parameters.getParameter<double>("etaECMin");
  etaECMax = parameters.getParameter<double>("etaECMax");
  etaOvlpMin = parameters.getParameter<double>("etaOvlpMin");
  etaOvlpMax = parameters.getParameter<double>("etaOvlpMax");

  int nBin = 0;
  for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
    if (iEtaRegion==0) { EtaName = "";         nBin = etaBin;} 
    if (iEtaRegion==1) { EtaName = "_Barrel";  nBin = etaBBin;}
    if (iEtaRegion==2) { EtaName = "_EndCap";  nBin = etaEBin;}
    if (iEtaRegion==3) { EtaName = "_Overlap"; nBin = etaOvlpBin;}

    GlbGlbMuon.push_back(dbe->book1D("GlbGlbMuon"+EtaName,"InvMass_{GLB,GLB}"+EtaName,nBin, 1.187, 201.187));
    GlbStaMuon.push_back(dbe->book1D("GlbStaMuon"+EtaName,"InvMass_{GLB,STA}"+EtaName,nBin, 1.187, 201.187));
    GlbTrkMuon.push_back(dbe->book1D("GlbTrkMuon"+EtaName,"InvMass_{GLB,TRK}"+EtaName,nBin, 1.187, 201.187));
    GlbGPTMuon.push_back(dbe->book1D("GlbGPTMuon"+EtaName,"InvMass_{GLB,GPT}"+EtaName,nBin, 1.187, 201.187));
 
    StaStaMuon.push_back(dbe->book1D("StaStaMuon"+EtaName,"InvMass_{STA,STA}"+EtaName,nBin, 1.187, 201.187));
    StaTrkMuon.push_back(dbe->book1D("StaTrkMuon"+EtaName,"InvMass_{STA,TRK}"+EtaName,nBin, 1.187, 201.187));
    StaGPTMuon.push_back(dbe->book1D("StaGPTMuon"+EtaName,"InvMass_{STA,GPT}"+EtaName,nBin, 1.187, 201.187));

    TrkTrkMuon.push_back(dbe->book1D("TrkTrkMuon"+EtaName,"InvMass_{TRK,TRK}"+EtaName,nBin, 1.187, 201.187));
    TrkGPTMuon.push_back(dbe->book1D("TrkGPTMuon"+EtaName,"InvMass_{TRK,GPT}"+EtaName,nBin, 1.187, 201.187));

    GPTGPTMuon.push_back(dbe->book1D("GPTGPTMuon"+EtaName,"InvMass_{GPT,GPT}"+EtaName,nBin, 1.187, 201.187 ));
  }
}
void DiMuonHistograms::analyze(const edm::Event & iEvent,const edm::EventSetup& iSetup) {

  LogTrace(metname)<<"[DiMuonHistograms] Analyze the mu in different eta regions";
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(theMuonCollectionLabel, muons);
  
  if(!muons.isValid()) return;

  // Loop on muon collection
  TLorentzVector Mu1[4], Mu2[4];
  bool Mu1Type[4][4] = {
    {false,false,false,false}, //Glb
    {false,false,false,false}, //Sta
    {false,false,false,false}, //Trk
    {false,false,false,false}  //GMPT
  };
  
  bool Mu2Type[4][4] = {
    {false,false,false,false}, //Glb
    {false,false,false,false}, //Sta
    {false,false,false,false}, //Trk
    {false,false,false,false}  //GMPT
  };
  float Mu1Charge[4] = {0.,0.,0.,0.};
  float Mu2Charge[4] = {0.,0.,0.,0.};
  for (reco::MuonCollection::const_iterator recoMu1 = muons->begin(); recoMu1!=muons->end(); ++recoMu1) {
    LogTrace(metname)<<"[DiMuonHistograms] loop over 1st muon"<<endl;
    for (int i=0; i<4; i++){
      for (int j=0; j<4; j++){
	Mu1Type[i][j] = false;
      }
    }
    
    if (recoMu1->isGlobalMuon()) {
      reco::TrackRef recoCombinedGlbTrack = recoMu1->combinedMuon();
      Mu1[0].SetPxPyPzE(recoCombinedGlbTrack->px(), recoCombinedGlbTrack->py(),
			recoCombinedGlbTrack->pz(), recoCombinedGlbTrack->p());
      for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
	if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	if (iEtaRegion==3) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 

	if(fabs(recoCombinedGlbTrack->eta())>EtaCutMin && fabs(recoCombinedGlbTrack->eta())<EtaCutMax){
	  Mu1Type[0][iEtaRegion] = true;
	}
      }
      Mu1Charge[0] = recoCombinedGlbTrack->charge();


      if (recoMu1->combinedMuon()->normalizedChi2()<10.	&& 
	  recoMu1->combinedMuon()->hitPattern().numberOfValidMuonHits() >0 ) {
	Mu1[3].SetPxPyPzE(recoCombinedGlbTrack->px(), recoCombinedGlbTrack->py(),
			  recoCombinedGlbTrack->pz(), recoCombinedGlbTrack->p());
	for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
	  if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	  if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	  if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	  if (iEtaRegion==3) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 

	  if(fabs(recoCombinedGlbTrack->eta())>EtaCutMin && fabs(recoCombinedGlbTrack->eta())<EtaCutMax){
	    Mu1Type[3][iEtaRegion] = true;
	  }
	}
	Mu1Charge[3] = recoCombinedGlbTrack->charge();
      }
    }
    
    if (recoMu1->isStandAloneMuon()) {
      reco::TrackRef recoStaTrack = recoMu1->standAloneMuon();
      Mu1[1].SetPxPyPzE(recoStaTrack->px(), recoStaTrack->py(),
			  recoStaTrack->pz(), recoStaTrack->p());
      for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
	if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	if (iEtaRegion==3) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 

	if(fabs(recoStaTrack->eta())>EtaCutMin && fabs(recoStaTrack->eta())<EtaCutMax){
	  Mu1Type[1][iEtaRegion] = true;
	}
      }
      Mu1Charge[1] = recoStaTrack->charge();
    }
    
    if (recoMu1->isTrackerMuon()) {
      reco::TrackRef recoTrack = recoMu1->track();
      Mu1[2].SetPxPyPzE(recoTrack->px(), recoTrack->py(),
			recoTrack->pz(), recoTrack->p());
      for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
	if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	if (iEtaRegion==3) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 

	if(fabs(recoTrack->eta())>EtaCutMin && fabs(recoTrack->eta())<EtaCutMax){
	  Mu1Type[2][iEtaRegion] = true;
	}
      }
      Mu1Charge[2] = recoTrack->charge();
    }
	

    // Loop on second muons to fill invariant mass plots
    for (reco::MuonCollection::const_iterator recoMu2 = recoMu1; recoMu2!=muons->end(); ++recoMu2){ 
      LogTrace(metname)<<"[DiMuonHistograms] loop over 2nd muon"<<endl;
      for (int i=0; i<4; i++){
	for (int j=0; j<4; j++){
	  Mu2Type[i][j] = false;
	}
      }

      if (recoMu1==recoMu2) continue;
      if (recoMu2->isGlobalMuon()) {
	reco::TrackRef recoCombinedGlbTrack = recoMu2->combinedMuon();
	Mu2[0].SetPxPyPzE(recoCombinedGlbTrack->px(), recoCombinedGlbTrack->py(),
			  recoCombinedGlbTrack->pz(), recoCombinedGlbTrack->p());
	for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
	  if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	  if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	  if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	  if (iEtaRegion==3) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 
	  
	  if(fabs(recoCombinedGlbTrack->eta())>EtaCutMin && fabs(recoCombinedGlbTrack->eta())<EtaCutMax){
	    Mu2Type[0][iEtaRegion] = true;
	  }
	}
	Mu2Charge[0] = recoCombinedGlbTrack->charge();

	if (recoMu2->combinedMuon()->normalizedChi2()<10.	&& 
	    recoMu2->combinedMuon()->hitPattern().numberOfValidMuonHits() >0 ) {
	  Mu2[3].SetPxPyPzE(recoCombinedGlbTrack->px(), recoCombinedGlbTrack->py(),
			    recoCombinedGlbTrack->pz(), recoCombinedGlbTrack->p());
	  for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
	    if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	    if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	    if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	    if (iEtaRegion==3) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 

	    if(fabs(recoCombinedGlbTrack->eta())>EtaCutMin && fabs(recoCombinedGlbTrack->eta())<EtaCutMax){
	      Mu2Type[3][iEtaRegion] = true;
	    }
	  }
	  Mu2Charge[3] = recoCombinedGlbTrack->charge();
	}
      }
    
      if (recoMu2->isStandAloneMuon()) {
	reco::TrackRef recoStaTrack = recoMu2->standAloneMuon();
	Mu2[1].SetPxPyPzE(recoStaTrack->px(), recoStaTrack->py(),
			    recoStaTrack->pz(), recoStaTrack->p());
	for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
	  if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	  if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	  if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	  if (iEtaRegion==3) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 
	  
	  if(fabs(recoStaTrack->eta())>EtaCutMin && fabs(recoStaTrack->eta())<EtaCutMax){
	    Mu2Type[1][iEtaRegion] = true;
	  }
	}
	Mu2Charge[1] = recoStaTrack->charge();
      }
    
      if (recoMu2->isTrackerMuon()) {
	reco::TrackRef recoTrack = recoMu2->track();
	Mu2[2].SetPxPyPzE(recoTrack->px(), recoTrack->py(),
			  recoTrack->pz(), recoTrack->p());
	for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
	  if(fabs(recoTrack->eta())>EtaCutMin && fabs(recoTrack->eta())<EtaCutMax){
	    Mu2Type[2][iEtaRegion] = true;
	  }
	}
	Mu2Charge[2] = recoTrack->charge();
      }
      
      
      LogTrace(metname)<<"[DiMuonHistograms] Calculate invmass"<<endl;
      float InvMass[4][4];
      for (unsigned int iEtaRegion=0; iEtaRegion<4; iEtaRegion++){
	if (iEtaRegion==0) {EtaCutMin= 0.;         EtaCutMax=2.4;       }
	if (iEtaRegion==1) {EtaCutMin= etaBMin;    EtaCutMax=etaBMax;   } 
	if (iEtaRegion==2) {EtaCutMin= etaECMin;   EtaCutMax=etaECMax;  }
	if (iEtaRegion==3) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 
	
	for (int mu1=0; mu1<4; mu1++){
	  for (int mu2=0; mu2<4; mu2++){
	    if (Mu1Type[mu1] && Mu1Type[mu2])
	      InvMass[mu1][mu2] = (Mu1[mu1]+Mu2[mu2]).M();
	    else
	      InvMass[mu1][mu2] = -99.;
	  }
	}
	
	LogTrace(metname)<<"[DiMuonHistograms] Fill  invmass histograms"<<endl;
	if (Mu1Type[0][iEtaRegion] && Mu1Type[0][iEtaRegion]) GlbGlbMuon[iEtaRegion]->Fill(InvMass[0][0]);
	if (Mu1Type[0][iEtaRegion] && Mu1Type[1][iEtaRegion]) GlbStaMuon[iEtaRegion]->Fill(InvMass[0][1]);
	if (Mu1Type[0][iEtaRegion] && Mu1Type[2][iEtaRegion]) GlbTrkMuon[iEtaRegion]->Fill(InvMass[0][2]);	    
	if (Mu1Type[0][iEtaRegion] && Mu1Type[3][iEtaRegion]) GlbGPTMuon[iEtaRegion]->Fill(InvMass[0][3]);	    
	  
	if (Mu1Type[1][iEtaRegion] && Mu1Type[1][iEtaRegion]) StaStaMuon[iEtaRegion]->Fill(InvMass[1][1]);
	if (Mu1Type[1][iEtaRegion] && Mu1Type[2][iEtaRegion]) StaTrkMuon[iEtaRegion]->Fill(InvMass[1][2]);	    
	if (Mu1Type[1][iEtaRegion] && Mu1Type[3][iEtaRegion]) StaGPTMuon[iEtaRegion]->Fill(InvMass[1][3]);	    
	
	if (Mu1Type[2][iEtaRegion] && Mu1Type[2][iEtaRegion]) TrkTrkMuon[iEtaRegion]->Fill(InvMass[2][2]);	    
	if (Mu1Type[2][iEtaRegion] && Mu1Type[3][iEtaRegion]) TrkGPTMuon[iEtaRegion]->Fill(InvMass[2][3]);	    

	if (Mu1Type[3][iEtaRegion] && Mu1Type[3][iEtaRegion]) GPTGPTMuon[iEtaRegion]->Fill(InvMass[3][3]);	    

      } //for each eta region
    } //muon2
  } //muon1
}

