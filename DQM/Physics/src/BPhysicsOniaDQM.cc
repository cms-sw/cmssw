/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/06/10 12:07:47 $
 *  $Revision: 1.0 $
 *  \author S. Bolognesi, Eric - CERN
 */

#include "DQM/Physics/src/BPhysicsOniaDQM.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <string>
#include <cmath>
using namespace std;
using namespace edm;
using namespace reco;



BPhysicsOniaDQM::BPhysicsOniaDQM(const ParameterSet& parameters) {

  // the services
  //theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));
  
  // Muon Collection Label
  theMuonCollectionLabel = parameters.getParameter<InputTag>("MuonCollection");
}

BPhysicsOniaDQM::~BPhysicsOniaDQM() { 
  

}


void BPhysicsOniaDQM::beginJob(EventSetup const& iSetup) {
 
  metname = "oniaAnalyzer";

  LogTrace(metname)<<"[BPhysicsOniaDQM] Parameters initialization";
  theDbe = Service<DQMStore>().operator->();
 
  theDbe->setCurrentFolder("Physics/BPhysics");  // Use folder with name of PAG
  global_background = theDbe->book1D("global_background", "global background", 750, 0, 15);
  diMuonMass_global = theDbe->book1D("diMuonMass_global", "dimuon mass", 750, 0, 15);
  tracker_background = theDbe->book1D("tracker_background", "tracker background", 750, 0, 15);
  diMuonMass_tracker = theDbe->book1D("diMuonMass_tracker", "dimuon mass", 750, 0, 15);
  standalone_background = theDbe->book1D("standalone_background", "standalone background", 500, 0, 15);
  diMuonMass_standalone = theDbe->book1D("diMuonMass_standalone", "dimuon mass", 500, 0, 15);


}


void BPhysicsOniaDQM::analyze(const Event& iEvent, const EventSetup& iSetup) {

  LogTrace(metname)<<"[BPhysicsOniaDQM] Analysis of event # ";
  
  // theService->update(iSetup);

  // Take the STA muon container
  Handle<MuonCollection> muons;
  iEvent.getByLabel(theMuonCollectionLabel,muons);

  if(muons.isValid()){
    pair<Muon,Muon> bestMassMuons;
    for (MuonCollection::const_iterator recoMu1 = muons->begin(); recoMu1!=muons->end(); ++recoMu1){
      
      // only loop over the remaining muons if recoMu1 is one of the following
      
      if(recoMu1->isGlobalMuon() || recoMu1->isTrackerMuon() || recoMu1->isStandAloneMuon()){
	for (MuonCollection::const_iterator recoMu2 = recoMu1+1; recoMu2!=muons->end(); ++recoMu2){
	  
	  // fill the relevant histograms if recoMu2 satisfies one of the following
	  if (recoMu1->isGlobalMuon() && recoMu2->isGlobalMuon()){
	    math::XYZVector vec1 = recoMu1->globalTrack()->momentum();
	    math::XYZVector vec2 = recoMu2->globalTrack()->momentum();
	    float massJPsi = computeMass(vec1,vec2);
	    
	    
	    // if opposite charges, fill _global, else fill _background
	    if (((*recoMu1).charge()*(*recoMu2).charge())<0) {
	      diMuonMass_global->Fill(massJPsi);
	    } else {
	      global_background->Fill (massJPsi);
	    }

	  }
	  if(recoMu1->isStandAloneMuon() && recoMu2->isStandAloneMuon()){
	    math::XYZVector vec1 = recoMu1->outerTrack()->momentum();
	    math::XYZVector vec2 = recoMu2->outerTrack()->momentum();
	    float massJPsi = computeMass(vec1,vec2);
	    
	    // if opposite charges, fill _standalone, else fill _background
	    if (((*recoMu1).charge()*(*recoMu2).charge())<0) {
	      diMuonMass_standalone->Fill(massJPsi);
	    } else {
	      standalone_background->Fill (massJPsi);
	    }
	  }
	  if(recoMu1->isTrackerMuon() && recoMu2->isTrackerMuon()){
	    math::XYZVector vec1 = recoMu1->innerTrack()->momentum();
	    math::XYZVector vec2 = recoMu2->innerTrack()->momentum();
	    float massJPsi = computeMass(vec1,vec2);
	    
	    // if opposite charges, fill _tracker, else fill _background
	    if (((*recoMu1).charge()*(*recoMu2).charge())<0) {
	      diMuonMass_tracker->Fill(massJPsi);
	    } else {
	      tracker_background->Fill (massJPsi);
	    }
	  }
	}
      }
    } 
  }    
}




void BPhysicsOniaDQM::endJob(void) {
  LogTrace(metname)<<"[BPhysicsOniaDQM] Saving the histos";
}


float BPhysicsOniaDQM::computeMass(const math::XYZVector &vec1,const math::XYZVector &vec2){
  // mass of muon
  float massMu = 0.10566;
  float eMu1 = sqrt(massMu*massMu + vec1.Mag2());
  float eMu2 = sqrt(massMu*massMu + vec2.Mag2());
  float pJPsi = sqrt((vec1+vec2).Mag2());
  float eJPsi = eMu1 + eMu2;
  float massJPsi = sqrt(eJPsi*eJPsi - pJPsi*pJPsi);
  return massJPsi;
}



