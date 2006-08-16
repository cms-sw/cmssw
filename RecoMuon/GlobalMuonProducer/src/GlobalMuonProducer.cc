/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a standalone reonstructed muon.
 *
 *   $Date: 2006/07/21 21:53:05 $
 *   $Revision: 1.10 $
 *
 *   \author  R.Bellan - INFN TO
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonProducer.h"

// TrackFinder and specific GLB Trajectory Builder
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"

// Input and output collection
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

using namespace edm;
using namespace std;

//
// constructor with config
//
GlobalMuonProducer::GlobalMuonProducer(const ParameterSet& parameterSet) {

  LogDebug("Muon|RecoMuon|GlobalMuonProducer") << "constructor called" << endl;

  // Parameter set for the Builder
  ParameterSet GLB_pSet = parameterSet.getParameter<ParameterSet>("GLBTrajBuilderParameters");

  // STA Muon Collection Label
  theSTACollectionLabel = parameterSet.getUntrackedParameter<string>("MuonCollectionLabel");

  // instantiate the concrete trajectory builder in the Track Finder
  GlobalMuonTrajectoryBuilder* gmtb = new GlobalMuonTrajectoryBuilder(GLB_pSet);
  theTrackFinder = new MuonTrackFinder(gmtb);
  
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();

  produces<reco::MuonCollection>();

}


//
// destructor
//
GlobalMuonProducer::~GlobalMuonProducer() {

  LogDebug("Muon|RecoMuon|GlobalMuonProducer") << "destructor called" << endl;
  if (theTrackFinder) delete theTrackFinder;

}


//
// reconstruct muons
//
void GlobalMuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const std::string metname = "Muon|RecoMuon|GlobalMuonProducer";  
  LogDebug(metname)<<endl<<endl<<endl;
  LogDebug(metname)<<"Global Muon Reconstruction started"<<endl;  
  
  // Take the STA muon container
  LogDebug(metname)<<"Taking the Stans Alone Muons: "<<theSTACollectionLabel<<endl; 
  Handle<reco::TrackCollection> staMuons;
  event.getByLabel(theSTACollectionLabel,staMuons);
  
  // Reconstruct the tracks in the tracker+muon system
  LogDebug(metname)<<"Track Reconstruction"<<endl;
  theTrackFinder->reconstruct(staMuons,event,eventSetup);
  
  LogDebug(metname)<<"Event loaded"
                   <<"================================"
                   <<endl<<endl;
    
}
