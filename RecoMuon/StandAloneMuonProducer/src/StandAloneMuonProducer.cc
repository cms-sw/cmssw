/**  \class StandAloneMuonProducer
 * 
 *   StandAlone muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *   $Date: 2008/10/06 14:03:43 $
 *   $Revision: 1.31 $
 *
 *   \author  R.Bellan - INFN TO
 */

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/StandAloneMuonProducer/src/StandAloneMuonProducer.h"

// TrackFinder and Specific STA Trajectory Builder
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/ExhaustiveMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/DirectMuonTrajectoryBuilder.h"

#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

// Input and output collection

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

#include <string>

using namespace edm;
using namespace std;

/// constructor with config
StandAloneMuonProducer::StandAloneMuonProducer(const ParameterSet& parameterSet){
  LogTrace("Muon|RecoMuon|StandAloneMuonProducer")<<"constructor called"<<endl;

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("STATrajBuilderParameters");
  
  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");
  
  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");
  
  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);

  MuonTrackLoader * trackLoader = new MuonTrackLoader(trackLoaderParameters,theService);
  MuonTrajectoryBuilder * trajectoryBuilder = 0;
  // instantiate the concrete trajectory builder in the Track Finder
  string typeOfBuilder = parameterSet.getParameter<string>("MuonTrajectoryBuilder");
  if(typeOfBuilder == "StandAloneMuonTrajectoryBuilder")
    trajectoryBuilder = new StandAloneMuonTrajectoryBuilder(trajectoryBuilderParameters,theService);
  else if(typeOfBuilder == "DirectMuonTrajectoryBuilder")
    trajectoryBuilder = new DirectMuonTrajectoryBuilder(trajectoryBuilderParameters,theService);
  else if(typeOfBuilder == "Exhaustive")
    trajectoryBuilder = new ExhaustiveMuonTrajectoryBuilder(trajectoryBuilderParameters,theService);
  else{
    LogWarning("Muon|RecoMuon|StandAloneMuonProducer") << "No Trajectory builder associated with "<<typeOfBuilder
						       << ". Falling down to the default (StandAloneMuonTrajectoryBuilder)";
     trajectoryBuilder = new StandAloneMuonTrajectoryBuilder(trajectoryBuilderParameters,theService);
  }
  theTrackFinder = new MuonTrackFinder(trajectoryBuilder, trackLoader);

  setAlias(parameterSet.getParameter<std::string>("@module_label"));
  
  produces<reco::TrackCollection>().setBranchAlias(theAlias + "Tracks");
  produces<reco::TrackCollection>("UpdatedAtVtx").setBranchAlias(theAlias + "UpdatedAtVtxTracks");
  produces<TrackingRecHitCollection>().setBranchAlias(theAlias + "RecHits");
  produces<reco::TrackExtraCollection>().setBranchAlias(theAlias + "TrackExtras");
  produces<reco::TrackToTrackMap>().setBranchAlias(theAlias + "TrackToTrackMap");
  
  produces<std::vector<Trajectory> >().setBranchAlias(theAlias + "Trajectories");
  produces<TrajTrackAssociationCollection>().setBranchAlias(theAlias + "TrajToTrackMap");
}
  
/// destructor
StandAloneMuonProducer::~StandAloneMuonProducer(){
  LogTrace("Muon|RecoMuon|StandAloneMuonProducer")<<"StandAloneMuonProducer destructor called"<<endl;
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;
}

/// reconstruct muons
void StandAloneMuonProducer::produce(Event& event, const EventSetup& eventSetup){
  const std::string metname = "Muon|RecoMuon|StandAloneMuonProducer";
  
  LogTrace(metname)<<endl<<endl<<endl;
  LogTrace(metname)<<"Stand Alone Muon Reconstruction Started"<<endl;

  // Take the seeds container
  LogTrace(metname)<<"Taking the seeds: "<<theSeedCollectionLabel.label()<<endl;
  Handle<View<TrajectorySeed> > seeds; 
  event.getByLabel(theSeedCollectionLabel,seeds);

  // Update the services
  theService->update(eventSetup);
  NavigationSetter setter(*theService->muonNavigationSchool());

  // Reconstruct 
  LogTrace(metname)<<"Track Reconstruction"<<endl;
  theTrackFinder->reconstruct(seeds,event);
 
  LogTrace(metname)<<"Event loaded"
		   <<"================================"
		   <<endl<<endl;
}

