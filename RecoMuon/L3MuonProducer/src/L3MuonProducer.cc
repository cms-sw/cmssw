/**  \class L3MuonProducer
 * 
 *   L3 muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a L2 reonstructed muon.
 *
 *   $Date: $
 *   $Revision: $
 *   \author  A. Everett - Purdue University
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L3MuonProducer/src/L3MuonProducer.h"

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
L3MuonProducer::L3MuonProducer(const ParameterSet& parameterSet) {

  LogDebug("L3MuonProducer") << "constructor called" << endl;

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("L3TrajBuilderParameters");

  // L2 Muon Collection Label
  theL2CollectionLabel = parameterSet.getParameter<InputTag>("MuonCollectionLabel");

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  // instantiate the concrete trajectory builder in the Track Finder
  GlobalMuonTrajectoryBuilder* gmtb = new GlobalMuonTrajectoryBuilder(trajectoryBuilderParameters, theService);
  theTrackFinder = new MuonTrackFinder(gmtb, new MuonTrackLoader(trackLoaderParameters, theService) );
  
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<vector<Trajectory> >() ;

  produces<reco::MuonCollection>();

}


//
// destructor
//
L3MuonProducer::~L3MuonProducer() {

  LogDebug("L3MuonProducer") << "destructor called" << endl;
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;

}


//
// reconstruct muons
//
void L3MuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const string metname = "L3MuonProducer";  
  LogDebug(metname)<<endl<<endl<<endl;
  LogDebug(metname)<<"L3 Muon Reconstruction started"<<endl;  

  typedef vector<Trajectory> TrajColl;

  // Update the services
  theService->update(eventSetup);

  // Take the L2 muon container(s)
  LogDebug(metname)<<"Taking the L2 Muons "<<theL2CollectionLabel.label()<<endl;

  Handle<reco::TrackCollection> L2Muons;
  event.getByLabel(theL2CollectionLabel,L2Muons);

  Handle<vector<Trajectory> > L2MuonsTraj;

  try {
    event.getByLabel(theL2CollectionLabel,L2MuonsTraj);      
    LogDebug(metname)<<"Track Reconstruction (tracks, trajs) "<< L2Muons.product()->size() << " " << L2MuonsTraj.product()->size() <<endl;
      
  } catch (...) {
    LogDebug(metname)<<"Track Reconstruction (L2Tracks)"<<endl;
  }

  theTrackFinder->reconstruct(L2Muons, L2MuonsTraj, event);      

  
  LogDebug(metname)<<"Event loaded"
                   <<"================================"
                   <<endl<<endl;
    
}
