/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a standalone reonstructed muon.
 *
 *   $Date: 2006/11/03 17:49:53 $
 *   $Revision: 1.18 $
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
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("GLBTrajBuilderParameters");

  // STA Muon Collection Label
  theSTACollectionLabel = parameterSet.getParameter<InputTag>("MuonCollectionLabel");

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
GlobalMuonProducer::~GlobalMuonProducer() {

  LogDebug("Muon|RecoMuon|GlobalMuonProducer") << "destructor called" << endl;
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;

}


//
// reconstruct muons
//
void GlobalMuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const string metname = "Muon|RecoMuon|GlobalMuonProducer";  
  LogDebug(metname)<<endl<<endl<<endl;
  LogDebug(metname)<<"Global Muon Reconstruction started"<<endl;  

  typedef vector<Trajectory> TrajColl;

  // Update the services
  theService->update(eventSetup);

  // Take the STA muon container(s)
  LogDebug(metname)<<"Taking the Stand Alone Muons "<<theSTACollectionLabel.label()<<endl;

  Handle<reco::TrackCollection> staMuons;
  event.getByLabel(theSTACollectionLabel,staMuons);

  Handle<vector<Trajectory> > staMuonsTraj;

  try {
    event.getByLabel(theSTACollectionLabel,staMuonsTraj);      
    LogDebug(metname)<<"Track Reconstruction (tracks, trajs) "<< staMuons.product()->size() << " " << staMuonsTraj.product()->size() <<endl;
      
  } catch (...) {
    LogDebug(metname)<<"Track Reconstruction (staTracks)"<<endl;
  }

  theTrackFinder->reconstruct(staMuons, staMuonsTraj, event);      

  
  LogDebug(metname)<<"Event loaded"
                   <<"================================"
                   <<endl<<endl;
    
}
