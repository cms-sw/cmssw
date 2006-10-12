/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a standalone reonstructed muon.
 *
 *   $Date: 2006/08/31 18:29:32 $
 *   $Revision: 1.13 $
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
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

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

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  // the propagator name for the track loader
  string trackLoaderPropagatorName = parameterSet.getParameter<string>("TrackLoaderPropagator");

  // instantiate the concrete trajectory builder in the Track Finder
  GlobalMuonTrajectoryBuilder* gmtb = new GlobalMuonTrajectoryBuilder(GLB_pSet,theService);
  theTrackFinder = new MuonTrackFinder(gmtb, new MuonTrackLoader(trackLoaderPropagatorName,theService) );
  
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
  if (theService) delete theService;
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
  
  // Update the services
  theService->update(eventSetup);
  
  // Reconstruct the tracks in the tracker+muon system
  LogDebug(metname)<<"Track Reconstruction"<<endl;
  theTrackFinder->reconstruct(staMuons,event);
  
  LogDebug(metname)<<"Event loaded"
                   <<"================================"
                   <<endl<<endl;
    
}
