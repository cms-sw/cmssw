/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a standalone reonstructed muon.
 *
 *   $Date: 2006/06/26 23:55:32 $
 *   $Revision: 1.6 $
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
  theSTACollectionLabel = parameterSet.getParameter<string>("MuonCollectionLabel");

  // instantiate the concrete trajectory builder in the Track Finder

  //theTrackFinder = new MuonTrackFinder(new GlobalMuonTrajectoryBuilder(GLB_pSet),
  //                                     new GlobalMuonTrackLoader());

  theTrackFinder = new MuonTrackFinder(new GlobalMuonTrajectoryBuilder(GLB_pSet));
  

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
void GlobalMuonProducer::produce(Event& event, const EventSetup& setup) {
  
  LogDebug("Muon|RecoMuon|GlobalMuonProducer") << "Global Muon Reconstruction started" << endl;  

  // Take the STA muon container
  Handle<TrackCollection> staMuons;
  event.getByLabel(theSTACollectionLabel,staMuons);

  // Reconstruct the tracks in the tracker+muon system
  theTrackFinder->reconstruct(staMuons,event,setup);

}
