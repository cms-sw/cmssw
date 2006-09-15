//-------------------------------------------------
//
/**  \class L2MuonProducer
 * 
 *   Level-2 muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from Level-1 trigger seeds.
 *
 *
 *   $Date: 2006/09/15 08:33:49 $
 *   $Revision: 1.12 $
 *
 *   \author  R.Bellan - INFN TO
 */
//
//--------------------------------------------------

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L2MuonProducer/src/L2MuonProducer.h"

// TrackFinder and Specific STA/L2 Trajectory Builder
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include <string>

using namespace edm;
using namespace std;

/// constructor with config
L2MuonProducer::L2MuonProducer(const ParameterSet& parameterSet){
  LogDebug("Muon|RecoMuon|L2MuonProducer")<<"constructor called"<<endl;

  // Parameter set for the Builder
  ParameterSet L2_pSet = parameterSet.getParameter<ParameterSet>("L2TrajBuilderParameters");

  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // the services
  theService = new MuonServiceProxy(serviceParameters);

  // the propagator name for the track loader
  string trackLoaderPropagatorName = parameterSet.getParameter<string>("TrackLoaderPropagatorName");

  // instantiate the concrete trajectory builder in the Track Finder
  theTrackFinder = new MuonTrackFinder(new StandAloneMuonTrajectoryBuilder(L2_pSet,theService),
				       new MuonTrackLoader(trackLoaderPropagatorName,theService));
  
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}
  
/// destructor
L2MuonProducer::~L2MuonProducer(){
  LogDebug("Muon|RecoMuon|L2eMuonProducer")<<"L2MuonProducer destructor called"<<endl;
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;
}


/// reconstruct muons
void L2MuonProducer::produce(Event& event, const EventSetup& eventSetup){
  
 const std::string metname = "Muon|RecoMuon|L2MuonProducer";
  
  LogDebug(metname)<<endl<<endl<<endl;
  LogDebug(metname)<<"L2 Muon Reconstruction Started"<<endl;
  
  // Take the seeds container
  LogDebug(metname)<<"Taking the seeds: "<<theSeedCollectionLabel.label()<<endl;
  Handle<TrajectorySeedCollection> seeds; 
  event.getByLabel(theSeedCollectionLabel,seeds);

  // Update the services
  theService->update(eventSetup);
  
  // Reconstruct 
  LogDebug(metname)<<"Track Reconstruction"<<endl;
  theTrackFinder->reconstruct(seeds,event);
  
  LogDebug(metname)<<"Event loaded"
		   <<"================================"
		   <<endl<<endl;
}

