//  \class MuScleFitPlotter
//  Plotter for simulated,generated and reco info of muons
//
//  $Date: 2009/01/08 17:03:38 $
//  $Revision: 1.4 $
//  \author  C.Mariotti, S.Bolognesi - INFN Torino / T.Dorigo, M.De Mattia - INFN Padova
//
// ----------------------------------------------------------------------------------

#include "MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitPlotter.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Histograms.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitUtils.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include <CLHEP/Vector/LorentzVector.h>

#include "TFile.h"
#include "TTree.h"
#include "TMinuit.h"
#include <vector>

using namespace std;
using namespace edm;
using namespace reco; // For AODSIM MC objects

// Constructor
// ----------
MuScleFitPlotter::MuScleFitPlotter(){
  outputFile = new TFile("genSimRecoPlots.root","RECREATE");
  fillHistoMap();
}

MuScleFitPlotter::~MuScleFitPlotter(){
  outputFile->cd();
  writeHistoMap();
}

// Find and store in histograms the generated resonance and muons
// --------------------------------------------------------------
void MuScleFitPlotter::fillGen1(Handle<GenParticleCollection> genParticles){
  bool prova = false;
  //Loop on generated particles
  pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> muFromRes; 

  for (GenParticleCollection::const_iterator mcIter=genParticles->begin(); mcIter!=genParticles->end(); ++mcIter ){
   //Check if it's a resonance
    if (mcIter->status()==2 && 
	(fabs(mcIter->pdgId())==23  || fabs(mcIter->pdgId())==443    || fabs(mcIter->pdgId())==100443 || 
	 fabs(mcIter->pdgId())==553 || fabs(mcIter->pdgId())==100553 || fabs(mcIter->pdgId())==200553)){
      mapHisto["hGenRes"]->Fill(mcIter->p4());
    }   
    //Check if it's a muon from a resonance
    if(mcIter->status()==1 && fabs(mcIter->pdgId())==13){
      int momPdgId = abs(mcIter->mother()->pdgId());
      if(momPdgId==23  || momPdgId==443    || momPdgId==100443 || 
	 momPdgId==553 || momPdgId==100553 || momPdgId==200553){
	mapHisto["hGenMu"]->Fill(mcIter->p4());     
	cout<<"genmu "<<mcIter->p4()<<endl;
	if(mcIter->charge()>0){
	  muFromRes.first = mcIter->p4();
	  prova = true;
	}
	else
	  muFromRes.second = mcIter->p4();
      }
    }
  }
  if(!prova)
    cout<<"hgenmumu not found"<<endl;
  cout<<"genmumu "<<muFromRes.first+muFromRes.second<<endl;
  
  mapHisto["hGenMuMu"]->Fill(muFromRes.first+muFromRes.second);
}

// Find and store in histograms the generated resonance and muons
// --------------------------------------------------------------
void MuScleFitPlotter::fillGen2(Handle<HepMCProduct> evtMC){
  
  //Loop on generated particles
  const HepMC::GenEvent* Evt = evtMC->GetEvent();
  pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> muFromRes; 

  for (HepMC::GenEvent::particle_const_iterator part=Evt->particles_begin(); 
       part!=Evt->particles_end(); part++) {
    //cout<<"PDG ID "<< (*part)->pdg_id() <<"    status "<< (*part)->status()
    //	<<"   pt "<<(*part)->momentum().perp()<< "     eta  "<<(*part)->momentum().eta()<<endl    ;
     //Check if it's a resonance	
    if ((*part)->status()==2 && 
	((*part)->pdg_id()==23  || (*part)->pdg_id()==443    || (*part)->pdg_id()==100443 ||
	 (*part)->pdg_id()==553 || (*part)->pdg_id()==100553 || (*part)->pdg_id()==200553)) {
      mapHisto["hGenRes"]->Fill(reco::Particle::LorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
							      (*part)->momentum().pz(),(*part)->momentum().e()));
   }
    //Check if it's a muon from a resonance
    if (fabs((*part)->pdg_id())==13 && (*part)->status()==1) {      
      bool fromRes=false;
      for (HepMC::GenVertex::particle_iterator mother = 
	     (*part)->production_vertex()->particles_begin(HepMC::ancestors);
	   mother != (*part)->production_vertex()->particles_end(HepMC::ancestors); ++mother) {
	if ((*mother)->pdg_id()==23  || (*mother)->pdg_id()==443    || (*mother)->pdg_id()==100443 || 
	    (*mother)->pdg_id()==553 || (*mother)->pdg_id()==100553 || (*mother)->pdg_id()==200553) {
	  fromRes=true;
	}
      }
      if(fromRes) {	
	mapHisto["hGenMu"]->Fill(reco::Particle::LorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
							       (*part)->momentum().pz(),(*part)->momentum().e()));
	if((*part)->pdg_id()==-13)
	  muFromRes.first = (reco::Particle::LorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
							   (*part)->momentum().pz(),(*part)->momentum().e()));
	else
	  muFromRes.second = (reco::Particle::LorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
							    (*part)->momentum().pz(),(*part)->momentum().e()));
      }
    }
  }
  mapHisto["hGenMuMu"]->Fill(muFromRes.first+muFromRes.second);
}

// Find and store in histograms the simulated resonance and muons
// --------------------------------------------------------------
 void MuScleFitPlotter::fillSim(Handle<SimTrackContainer> simTracks){

   vector<SimTrack> simMuons;
   int numberOfSimMuonsAcc=0;

   //Loop on simulated tracks
   for (SimTrackContainer::const_iterator simTrack=simTracks->begin(); simTrack!=simTracks->end(); ++simTrack) {
     // Select the muons from all the simulated tracks
     if (fabs((*simTrack).type())==13) {
       simMuons.push_back(*simTrack);	  
       mapHisto["hSimMu"]->Fill((*simTrack).momentum());
       if ((fabs((*simTrack).momentum().eta())<2.5) && ((*simTrack).momentum().pt()>2.5)){
	 mapHisto["hSimMu_Acc"]->Fill((*simTrack).momentum());
	 numberOfSimMuonsAcc++;
       }
     }
   }
   mapHisto["hSimMu"]->Fill(simMuons.size());
   mapHisto["hSimMu_Acc"]->Fill(numberOfSimMuonsAcc);

   // Recombine all the possible Z from simulated muons
   if (simMuons.size()>=2) {
     for (vector<SimTrack>::const_iterator  imu=simMuons.begin(); imu != simMuons.end(); ++imu) {   
       for (vector<SimTrack>::const_iterator imu2=imu+1; imu2!=simMuons.end(); ++imu2) {
	 if (imu==imu2) continue;
	    
	 // Try all the pairs with opposite charge
	 if (((*imu).charge()*(*imu2).charge())<0) {
	   reco::Particle::LorentzVector Z = (*imu).momentum()+(*imu2).momentum();
	   mapHisto["hSimMuPMuM"]->Fill(Z); 
	   if (fabs((*imu).momentum().eta())<2.5 && fabs((*imu2).momentum().eta())<2.5
	       && (*imu).momentum().pt()>2.5 && (*imu2).momentum().pt()>2.5) {
	     mapHisto["hSimMuPMuM_Acc"]->Fill(Z);
	   }
	 }
       }
     }
   
    // Plots for the best possible simulated resonance
     pair<SimTrack,SimTrack> simMuFromBestRes = MuScleFitUtils::findBestSimuRes(simMuons);
     reco::Particle::LorentzVector bestSimZ = (simMuFromBestRes.first).momentum()+(simMuFromBestRes.second).momentum();
     mapHisto["hSimBestRes"]->Fill(bestSimZ);
     if (fabs(simMuFromBestRes.first.momentum().eta())<2.5 && fabs(simMuFromBestRes.second.momentum().eta())<2.5 &&
	 simMuFromBestRes.first.momentum().pt()>2.5 && simMuFromBestRes.second.momentum().pt()>2.5) {
       mapHisto["hSimBestRes_Acc"]->Fill(bestSimZ);
       mapHisto["hSimBestResVSMu"]->Fill (simMuFromBestRes.first.momentum(), bestSimZ, int(simMuFromBestRes.first.charge()));
       mapHisto["hSimBestResVSMu"]->Fill (simMuFromBestRes.second.momentum(),bestSimZ, int(simMuFromBestRes.second.charge()));
    }
   }  
 }

// Find and store in histograms the RIGHT simulated resonance and muons
// --------------------------------------------------------------
 void MuScleFitPlotter::fillGenSim(Handle<HepMCProduct> evtMC, Handle<SimTrackContainer> simTracks){
   pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> simMuFromRes = 
     MuScleFitUtils::findSimMuFromRes(evtMC,simTracks);
   //Fill resonance info
   reco::Particle::LorentzVector rightSimRes = (simMuFromRes.first)+(simMuFromRes.second);
   mapHisto["hSimRightRes"]->Fill(rightSimRes);
   if ((fabs(simMuFromRes.first.Eta())<2.5 && fabs(simMuFromRes.second.Eta())<2.5) 
       && simMuFromRes.first.Pt()>2.5 && simMuFromRes.second.Pt()>2.5) {
     mapHisto["hSimRightRes_Acc"]->Fill(rightSimRes);
   }
 }


// Find and store in histograms the reconstructed resonance and muons
// --------------------------------------------------------------
 void MuScleFitPlotter::fillRec(vector<reco::LeafCandidate>& muons){
   for(vector<reco::LeafCandidate>::const_iterator mu1 = muons.begin(); mu1!=muons.end(); mu1++){
     mapHisto["hRecMu"]->Fill(mu1->p4());
     if (fabs(mu1->p4().eta())<2.5 && (mu1->p4().pt()>3.0)) {
	mapHisto["hRecMu_Acc"]->Fill(mu1->p4());
      }
     for(vector<reco::LeafCandidate>::const_iterator mu2 = muons.begin(); mu2!=muons.end(); mu2++){  
       if (mu1==mu2) continue;
       reco::Particle::LorentzVector Res (mu1->p4()+mu2->p4());
       mapHisto["hRecMuPMuM"]->Fill(Res);	  
       if (fabs(mu1->p4().eta())<2.5 && (mu1->p4().pt()>2.5) && 
	   fabs(mu2->p4().eta())<2.5 && (mu2->p4().pt()>2.5)){
	 mapHisto["hRecMuPMuM_Acc"]->Fill(Res);
       }
    } 
   }
 }


// Histogram booking
// -----------------
void MuScleFitPlotter::fillHistoMap() {

  // Generated Z and muons
  // ---------------------
  mapHisto["hGenRes"]         = new HParticle   ("hGenRes");
  mapHisto["hGenMu"]        = new HParticle   ("hGenMu");
  mapHisto["hGenMuMu"]      = new HParticle   ("hGenMuMu");

  // Simulated resonance and muons
  // -----------------------------
  mapHisto["hSimMu"]      = new HParticle ("hSimMu");
  mapHisto["hSimMu_Acc"]  = new HParticle ("hSimMu_Acc");

  mapHisto["hSimMuPMuM"]      = new HParticle ("hSimMuPMuM");      
  mapHisto["hSimMuPMuM_Acc"]  = new HParticle ("hSimMuPMuM_Acc"); 
                                                                 
  mapHisto["hSimBestMu"]      = new HParticle ("hSimBestMu");
  mapHisto["hSimBestMu_Acc"]  = new HParticle ("hSimBestMu_Acc");
  mapHisto["hSimBestRes"]         = new HParticle  ("hSimBestRes");
  mapHisto["hSimBestRes_Acc"]     = new HParticle  ("hSimBestRes_Acc"); 
  mapHisto["hSimBestResVSMu"]  = new HMassVSPart ("hSimBestResVSMu");
    
  mapHisto["hSimRightRes"]         = new HParticle  ("hSimRightZ");
  mapHisto["hSimRightRes_Acc"]     = new HParticle  ("hSimRightZ_Acc");
 
  // Reconstructed resonance and muons
  // -----------------------------  
  mapHisto["hRecMu"]      = new HParticle ("hRecMu");
  mapHisto["hRecMu_Acc"]  = new HParticle ("hRecMu_Acc");
  mapHisto["hRecMuPMuM"]         = new HParticle  ("hRecMuPMuM");
  mapHisto["hRecMuPMuM_Acc"]     = new HParticle  ("hRecMuPMuM_Acc");
}  


// Histogram saving
// -----------------
void MuScleFitPlotter::writeHistoMap() {
  outputFile->cd();
  for (map<string, Histograms*>::const_iterator histo=mapHisto.begin(); 
       histo!=mapHisto.end(); histo++) {
    (*histo).second->Write();
  }
}

