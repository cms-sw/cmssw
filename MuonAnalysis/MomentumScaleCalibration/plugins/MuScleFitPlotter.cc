//  \class MuScleFitPlotter
//  Plotter for simulated,generated and reco info of muons
//
//  $Date: 2012/12/20 16:09:22 $
//  $Revision: 1.6 $
//  \author  C.Mariotti, S.Bolognesi - INFN Torino / T.Dorigo, M.De Mattia - INFN Padova
//
// ----------------------------------------------------------------------------------

#include "MuScleFitPlotter.h"
#include "Histograms.h"
#include "MuScleFitUtils.h"
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

// Constructor
// ----------
MuScleFitPlotter::MuScleFitPlotter(std::string theGenInfoRootFileName){
  outputFile = new TFile(theGenInfoRootFileName.c_str(),"RECREATE");
  fillHistoMap();
}

MuScleFitPlotter::~MuScleFitPlotter(){
  outputFile->cd();
  writeHistoMap();
  outputFile->Close();
}

// Find and store in histograms the generated resonance and muons
// --------------------------------------------------------------
void MuScleFitPlotter::fillGen(const reco::GenParticleCollection* genParticles, bool PATmuons)
{
  //  bool prova = false;
  //Loop on generated particles
  std::pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> muFromRes;
  reco::Particle::LorentzVector genRes;

  int mothersFound[] = {0, 0, 0, 0, 0, 0};

  for( reco::GenParticleCollection::const_iterator mcIter=genParticles->begin(); mcIter!=genParticles->end(); ++mcIter ) {
    int status = mcIter->status();
    int pdgId = std::abs(mcIter->pdgId());
    //Check if it's a resonance
    if( status == 2 &&
        ( pdgId==23  || pdgId==443    || pdgId==100443 ||
          pdgId==553 || pdgId==100553 || pdgId==200553 ) ) {
      genRes = mcIter->p4();
      // std::cout << "mother's mother = " << mcIter->mother()->pdgId() << std::endl;
      if( pdgId == 23 ) mapHisto["hGenResZ"]->Fill(genRes);
      else if( pdgId == 443 || pdgId == 100443 ) mapHisto["hGenResJPsi"]->Fill(genRes);
      else if( pdgId == 553 || pdgId == 100553 || pdgId == 200553 ) mapHisto["hGenResUpsilon1S"]->Fill(genRes);
    }
    //Check if it's a muon from a resonance
    if( status==1 && pdgId==13 && !PATmuons) {
      int momPdgId = std::abs(mcIter->mother()->pdgId());
      if( momPdgId==23  || momPdgId==443    || momPdgId==100443 || 
          momPdgId==553 || momPdgId==100553 || momPdgId==200553 ) {
        if( momPdgId == 23 ) mothersFound[0] = 1;
        if( momPdgId == 443 || momPdgId == 100443 ) mothersFound[5] = 1;
        if( momPdgId == 553 || momPdgId == 100553 || momPdgId == 200553 ) mothersFound[3] = 1;
	mapHisto["hGenMu"]->Fill(mcIter->p4());
	std::cout<<"genmu "<<mcIter->p4()<<std::endl;
	if(mcIter->charge()>0){
	  muFromRes.first = mcIter->p4();
	  // prova = true;
	}
	else muFromRes.second = mcIter->p4();
      }
    }//if PATmuons you don't have the info of the mother !!! Here I assume is a JPsi
    if( status==1 && pdgId==13 && PATmuons) {
      mothersFound[5] = 1;
      mapHisto["hGenMu"]->Fill(mcIter->p4());
      std::cout<<"genmu "<<mcIter->p4()<<std::endl;
      if(mcIter->charge()>0){
	muFromRes.first = mcIter->p4();
	// prova = true;
      }
      else muFromRes.second = mcIter->p4();
    }
  }
  //   if(!prova)
  //     std::cout<<"hgenmumu not found"<<std::endl;

  if( mothersFound[0] == 1 ) {
    mapHisto["hGenMuMuZ"]->Fill(muFromRes.first+muFromRes.second);
    mapHisto["hGenResVSMuZ"]->Fill( muFromRes.first, genRes, 1 );
    mapHisto["hGenResVSMuZ"]->Fill( muFromRes.second,genRes, -1 );
  }
  if( mothersFound[3] == 1 ) {
    mapHisto["hGenMuMuUpsilon1S"]->Fill(muFromRes.first+muFromRes.second);
    mapHisto["hGenResVSMuUpsilon1S"]->Fill( muFromRes.first, genRes, 1 );
    mapHisto["hGenResVSMuUpsilon1S"]->Fill( muFromRes.second,genRes, -1 );
  }
  if( mothersFound[5] == 1 ) {
    mapHisto["hGenMuMuJPsi"]->Fill(muFromRes.first+muFromRes.second);
    mapHisto["hGenResVSMuJPsi"]->Fill( muFromRes.first, genRes, 1 );
    mapHisto["hGenResVSMuJPsi"]->Fill( muFromRes.second,genRes, -1 );
  }

  mapHisto["hGenResVsSelf"]->Fill( genRes, genRes, 1 );
}

// Find and store in histograms the generated resonance and muons
// --------------------------------------------------------------
void MuScleFitPlotter::fillGen(const edm::HepMCProduct* evtMC, bool sherpaFlag_)
{
  //Loop on generated particles
  const HepMC::GenEvent* Evt = evtMC->GetEvent();
  std::pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> muFromRes; 
  reco::Particle::LorentzVector genRes;
  
  int mothersFound[] = {0, 0, 0, 0, 0, 0};
  
  if( sherpaFlag_ ) {
    
    for (HepMC::GenEvent::particle_const_iterator part=Evt->particles_begin(); 
	 part!=Evt->particles_end(); part++) {
      if (fabs((*part)->pdg_id())==13 && (*part)->status()==1) {//looks for muon in the final state
	bool fromRes = false;
	for (HepMC::GenVertex::particle_iterator mother = (*part)->production_vertex()->particles_begin(HepMC::ancestors);//loops on the mother of the final state muons
	     mother != (*part)->production_vertex()->particles_end(HepMC::ancestors); ++mother) {
	  unsigned int motherPdgId = (*mother)->pdg_id();
          if( motherPdgId == 13 && (*mother)->status() == 3 ) fromRes = true;
	}
	if(fromRes){
	  if((*part)->pdg_id()==13){
	    muFromRes.first = (lorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
					     (*part)->momentum().pz(),(*part)->momentum().e()));
	  }
	  else if((*part)->pdg_id()==-13){
	    muFromRes.second = (lorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
					      (*part)->momentum().pz(),(*part)->momentum().e()));	    
	  }
	}
      }
      
    }
    mapHisto["hGenResZ"]->Fill(muFromRes.first+muFromRes.second);   
  }
  else{
    for (HepMC::GenEvent::particle_const_iterator part=Evt->particles_begin(); 
	 part!=Evt->particles_end(); part++) {
      int status = (*part)->status();
      int pdgId = std::abs((*part)->pdg_id());
      //std::cout<<"PDG ID "<< (*part)->pdg_id() <<"    status "<< (*part)->status()
      //<<"   pt "<<(*part)->momentum().perp()<< "     eta  "<<(*part)->momentum().eta()<<std::endl    ;
      //Check if it's a resonance	
      // For sherpa the resonance is not saved. The muons from the resonance can be identified
      // by having as mother a muon of status 3.
      
      if (pdgId==13 && status==1) {  
	if( status==2 && 
	    ( pdgId==23  || pdgId==443    || pdgId==100443 ||
	      pdgId==553 || pdgId==100553 || pdgId==200553 ) ) {
	  genRes = reco::Particle::LorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
						 (*part)->momentum().pz(),(*part)->momentum().e());
	  
	  if( pdgId == 23 ) mapHisto["hGenResZ"]->Fill(genRes);
	  if( pdgId == 443 ) mapHisto["hGenResJPsi"]->Fill(genRes);
	  if( pdgId == 553 ) {
	    // std::cout << "genRes mass = " << CLHEP::HepLorentzVector(genRes.x(),genRes.y(),genRes.z(),genRes.t()).m() << std::endl;
	    mapHisto["hGenResUpsilon1S"]->Fill(genRes);
	  }
	}
      
	//Check if it's a muon from a resonance
	if (pdgId==13 && status==1) {      
	  bool fromRes=false;
	  for (HepMC::GenVertex::particle_iterator mother = 
		 (*part)->production_vertex()->particles_begin(HepMC::ancestors);
	       mother != (*part)->production_vertex()->particles_end(HepMC::ancestors); ++mother) {
	    int motherPdgId = (*mother)->pdg_id();
	    if (motherPdgId==23  || motherPdgId==443    || motherPdgId==100443 || 
		motherPdgId==553 || motherPdgId==100553 || motherPdgId==200553) {
	      fromRes=true;
	      if( motherPdgId == 23 ) mothersFound[0] = 1;
	      if( motherPdgId == 443 ) mothersFound[3] = 1;
	      if( motherPdgId == 553 ) mothersFound[5] = 1;
	    }
	  }
	  
	  if(fromRes) {	
	    mapHisto["hGenMu"]->Fill(reco::Particle::LorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
								   (*part)->momentum().pz(),(*part)->momentum().e()));
	    mapHisto["hGenMuVSEta"]->Fill(reco::Particle::LorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
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
    }
  }
  if( mothersFound[0] == 1 ) {
    mapHisto["hGenMuMuZ"]->Fill(muFromRes.first+muFromRes.second);
    mapHisto["hGenResVSMuZ"]->Fill( muFromRes.first, genRes, 1 );
    mapHisto["hGenResVSMuZ"]->Fill( muFromRes.second,genRes, -1 );
  }
  if( mothersFound[3] == 1 ) {
    mapHisto["hGenMuMuUpsilon1S"]->Fill(muFromRes.first+muFromRes.second);
    mapHisto["hGenResVSMuUpsilon1S"]->Fill( muFromRes.first, genRes, 1 );
    mapHisto["hGenResVSMuUpsilon1S"]->Fill( muFromRes.second,genRes, -1 );
  }
  if( mothersFound[5] == 1 ) {
    mapHisto["hGenMuMuJPsi"]->Fill(muFromRes.first+muFromRes.second);
    mapHisto["hGenResVSMuJPsi"]->Fill( muFromRes.first, genRes, 1 );
    mapHisto["hGenResVSMuJPsi"]->Fill( muFromRes.second,genRes, -1 );
  }
  mapHisto["hGenResVsSelf"]->Fill( genRes, genRes, 1 );
}

// Find and store in histograms the simulated resonance and muons
// --------------------------------------------------------------
void MuScleFitPlotter::fillSim(edm::Handle<edm::SimTrackContainer> simTracks)
{
  std::vector<SimTrack> simMuons;

  //Loop on simulated tracks
  for( edm::SimTrackContainer::const_iterator simTrack=simTracks->begin(); simTrack!=simTracks->end(); ++simTrack ) {
    // Select the muons from all the simulated tracks
    if (fabs((*simTrack).type())==13) {
      simMuons.push_back(*simTrack);	  
      mapHisto["hSimMu"]->Fill((*simTrack).momentum());
    }
  }
  mapHisto["hSimMu"]->Fill(simMuons.size());

  // Recombine all the possible Z from simulated muons
  if( simMuons.size() >= 2 ) {
    for( std::vector<SimTrack>::const_iterator  imu=simMuons.begin(); imu != simMuons.end(); ++imu ) {   
      for( std::vector<SimTrack>::const_iterator imu2=imu+1; imu2!=simMuons.end(); ++imu2 ) {
	if (imu==imu2) continue;

	// Try all the pairs with opposite charge
	if (((*imu).charge()*(*imu2).charge())<0) {
	  reco::Particle::LorentzVector Z = (*imu).momentum()+(*imu2).momentum();
	  mapHisto["hSimMuPMuM"]->Fill(Z); 
	}
      }
    }

    // Plots for the best possible simulated resonance
    std::pair<SimTrack,SimTrack> simMuFromBestRes = MuScleFitUtils::findBestSimuRes(simMuons);
    reco::Particle::LorentzVector bestSimZ = (simMuFromBestRes.first).momentum()+(simMuFromBestRes.second).momentum();
    mapHisto["hSimBestRes"]->Fill(bestSimZ);
    if (fabs(simMuFromBestRes.first.momentum().eta())<2.5 && fabs(simMuFromBestRes.second.momentum().eta())<2.5 &&
	simMuFromBestRes.first.momentum().pt()>2.5 && simMuFromBestRes.second.momentum().pt()>2.5) {
      mapHisto["hSimBestResVSMu"]->Fill (simMuFromBestRes.first.momentum(), bestSimZ, int(simMuFromBestRes.first.charge()));
      mapHisto["hSimBestResVSMu"]->Fill (simMuFromBestRes.second.momentum(),bestSimZ, int(simMuFromBestRes.second.charge()));
    }
  }
}

// Find and store in histograms the RIGHT simulated resonance and muons
// --------------------------------------------------------------
void MuScleFitPlotter::fillGenSim(edm::Handle<edm::HepMCProduct> evtMC, edm::Handle<edm::SimTrackContainer> simTracks)
{
  std::pair<reco::Particle::LorentzVector, reco::Particle::LorentzVector> simMuFromRes = 
    MuScleFitUtils::findSimMuFromRes(evtMC,simTracks);
  //Fill resonance info
  reco::Particle::LorentzVector rightSimRes = (simMuFromRes.first)+(simMuFromRes.second);
  mapHisto["hSimRightRes"]->Fill(rightSimRes);
  /*if ((fabs(simMuFromRes.first.Eta())<2.5 && fabs(simMuFromRes.second.Eta())<2.5) 
    && simMuFromRes.first.Pt()>2.5 && simMuFromRes.second.Pt()>2.5) {
  }*/
}

// Find and store in histograms the reconstructed resonance and muons
// --------------------------------------------------------------
void MuScleFitPlotter::fillRec(std::vector<reco::LeafCandidate>& muons)
{
  for(std::vector<reco::LeafCandidate>::const_iterator mu1 = muons.begin(); mu1!=muons.end(); mu1++){
    mapHisto["hRecMu"]->Fill(mu1->p4());
    mapHisto["hRecMuVSEta"]->Fill(mu1->p4());
    for(std::vector<reco::LeafCandidate>::const_iterator mu2 = muons.begin(); mu2!=muons.end(); mu2++){  
      if (mu1->charge()<0 || mu2->charge()>0)
	continue;
      reco::Particle::LorentzVector Res (mu1->p4()+mu2->p4());
      mapHisto["hRecMuPMuM"]->Fill(Res);	  
    } 
  }
  mapHisto["hRecMu"]->Fill(muons.size());
}

/// Used when running on the root tree containing preselected muon pairs
void MuScleFitPlotter::fillTreeRec( const std::vector<std::pair<reco::Particle::LorentzVector, reco::Particle::LorentzVector> > & savedPairs )
{
  std::vector<std::pair<reco::Particle::LorentzVector, reco::Particle::LorentzVector> >::const_iterator muonPair = savedPairs.begin();
  for( ; muonPair != savedPairs.end(); ++muonPair ) {
    mapHisto["hRecMu"]->Fill(muonPair->first);
    mapHisto["hRecMuVSEta"]->Fill(muonPair->first);
    mapHisto["hRecMu"]->Fill(muonPair->second);
    mapHisto["hRecMuVSEta"]->Fill(muonPair->second);
    reco::Particle::LorentzVector Res( muonPair->first+muonPair->second );
    mapHisto["hRecMuPMuM"]->Fill(Res);
    mapHisto["hRecMu"]->Fill(savedPairs.size());
  }
}

/**
 * Used when running on the root tree and there is genInfo. <br>
 * ATTENTION: since we do not have any id information when reading from the root tree, we always
 * fill the Z histograms by default.
 */
void MuScleFitPlotter::fillTreeGen( const std::vector<std::pair<reco::Particle::LorentzVector, reco::Particle::LorentzVector> > & genPairs )
{
  std::vector<std::pair<reco::Particle::LorentzVector, reco::Particle::LorentzVector> >::const_iterator genPair = genPairs.begin();
  for( ; genPair != genPairs.end(); ++genPair ) {
    reco::Particle::LorentzVector genRes(genPair->first+genPair->second);
    mapHisto["hGenResZ"]->Fill(genRes);
    mapHisto["hGenMu"]->Fill(genPair->first);
    mapHisto["hGenMuVSEta"]->Fill(genPair->first);
    mapHisto["hGenMuMuZ"]->Fill(genRes);
    mapHisto["hGenResVSMuZ"]->Fill( genPair->first, genRes, 1 );
    mapHisto["hGenResVSMuZ"]->Fill( genPair->second, genRes, -1 );
    mapHisto["hGenMuMuUpsilon1S"]->Fill(genRes);
    mapHisto["hGenResVSMuUpsilon1S"]->Fill( genPair->first, genRes, 1 );
    mapHisto["hGenResVSMuUpsilon1S"]->Fill( genPair->second, genRes, -1 );
    mapHisto["hGenMuMuJPsi"]->Fill(genRes);
    mapHisto["hGenResVSMuJPsi"]->Fill( genPair->first, genRes, 1 );
    mapHisto["hGenResVSMuJPsi"]->Fill( genPair->second, genRes, -1 );
    mapHisto["hGenResVsSelf"]->Fill( genRes, genRes, 1 );
  }
}

// Histogram booking
// -----------------
void MuScleFitPlotter::fillHistoMap() {

  // Generated Z and muons
  // ---------------------
  mapHisto["hGenResJPsi"]      = new HParticle   ("hGenResJPsi", 3., 3.1);
  mapHisto["hGenResUpsilon1S"] = new HParticle   ("hGenResUpsilon1S", 9., 11.);
  mapHisto["hGenResZ"]         = new HParticle   ("hGenResZ", 60., 120.);
  mapHisto["hGenMu"]      = new HParticle  ("hGenMu");
  mapHisto["hGenMuVSEta"] = new HPartVSEta ("hGenMuVSEta");

  mapHisto["hGenMuMuJPsi"]      = new HParticle   ("hGenMuMuJPsi",3., 3.1 );
  mapHisto["hGenResVSMuJPsi"]   = new HMassVSPart ("hGenResVSMuJPsi",3., 3.1);
  mapHisto["hGenMuMuUpsilon1S"]      = new HParticle   ("hGenMuMuUpsilon1S", 9., 11.);
  mapHisto["hGenResVSMuUpsilon1S"]   = new HMassVSPart ("hGenResVSMuUpsilon1S", 9., 11.);
  mapHisto["hGenMuMuZ"]      = new HParticle   ("hGenMuMuZ", 60., 120.);
  mapHisto["hGenResVSMuZ"]   = new HMassVSPart ("hGenResVSMuZ", 60., 120.);

  mapHisto["hGenResVsSelf"] = new HMassVSPart ("hGenResVsSelf");

  // Simulated resonance and muons
  // -----------------------------
  mapHisto["hSimMu"]      = new HParticle ("hSimMu");

  mapHisto["hSimMuPMuM"]      = new HParticle ("hSimMuPMuM");      
                                                                 
  mapHisto["hSimBestMu"]      = new HParticle ("hSimBestMu");
  mapHisto["hSimBestRes"]         = new HParticle  ("hSimBestRes");
  mapHisto["hSimBestResVSMu"]  = new HMassVSPart ("hSimBestResVSMu");
    
  mapHisto["hSimRightRes"]         = new HParticle  ("hSimRightZ");
 
  // Reconstructed resonance and muons
  // -----------------------------  
  mapHisto["hRecMu"]      = new HParticle ("hRecMu");
  mapHisto["hRecMuVSEta"]      = new HPartVSEta ("hRecMuVSEta");
  mapHisto["hRecMuPMuM"]         = new HParticle  ("hRecMuPMuM");
}  


// Histogram saving
// -----------------
void MuScleFitPlotter::writeHistoMap() {
  outputFile->cd();
  for (std::map<std::string, Histograms*>::const_iterator histo=mapHisto.begin(); 
       histo!=mapHisto.end(); histo++) {
    (*histo).second->Write();
  }
}

