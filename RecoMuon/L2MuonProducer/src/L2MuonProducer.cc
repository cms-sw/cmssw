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
 *
 *   \author  R.Bellan - INFN TO
 */
//
//--------------------------------------------------

#include "RecoMuon/L2MuonProducer/src/L2MuonProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// TrackFinder and Specific STA/L2 Trajectory Builder
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/ExhaustiveMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"

#include <string>

using namespace edm;
using namespace std;

/// constructor with config
L2MuonProducer::L2MuonProducer(const ParameterSet& parameterSet){
  LogTrace("Muon|RecoMuon|L2MuonProducer")<<"constructor called"<<endl;

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("L2TrajBuilderParameters");

  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");
  seedsToken = consumes<edm::View<TrajectorySeed> >(theSeedCollectionLabel);
  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");

  // the services
  theService = new MuonServiceProxy(serviceParameters);

  MuonTrajectoryBuilder * trajectoryBuilder = 0;
  // instantiate the concrete trajectory builder in the Track Finder
  edm::ConsumesCollector  iC = consumesCollector();
  string typeOfBuilder = parameterSet.existsAs<string>("MuonTrajectoryBuilder") ? 
    parameterSet.getParameter<string>("MuonTrajectoryBuilder") : "StandAloneMuonTrajectoryBuilder";
  if(typeOfBuilder == "StandAloneMuonTrajectoryBuilder" || typeOfBuilder == "")
    trajectoryBuilder = new StandAloneMuonTrajectoryBuilder(trajectoryBuilderParameters,theService,iC);
  else if(typeOfBuilder == "Exhaustive")
    trajectoryBuilder = new ExhaustiveMuonTrajectoryBuilder(trajectoryBuilderParameters,theService,iC);
  else{
    LogWarning("Muon|RecoMuon|StandAloneMuonProducer") << "No Trajectory builder associated with "<<typeOfBuilder
    						       << ". Falling down to the default (StandAloneMuonTrajectoryBuilder)";
    trajectoryBuilder = new StandAloneMuonTrajectoryBuilder(trajectoryBuilderParameters,theService,iC);
  }
  theTrackFinder = new MuonTrackFinder(trajectoryBuilder,
				       new MuonTrackLoader(trackLoaderParameters, iC, theService),
				       new MuonTrajectoryCleaner(true));
  
  produces<reco::TrackCollection>();
  produces<reco::TrackCollection>("UpdatedAtVtx");
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<reco::TrackToTrackMap>();

  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();

  produces<edm::AssociationMap<edm::OneToMany<std::vector<L2MuonTrajectorySeed>, std::vector<L2MuonTrajectorySeed> > > >();
}
  
/// destructor
L2MuonProducer::~L2MuonProducer(){
  LogTrace("Muon|RecoMuon|L2eMuonProducer")<<"L2MuonProducer destructor called"<<endl;
  delete theService;
  delete theTrackFinder;
}


/// reconstruct muons
void L2MuonProducer::produce(Event& event, const EventSetup& eventSetup){
  
 const std::string metname = "Muon|RecoMuon|L2MuonProducer";
  
  LogTrace(metname)<<endl<<endl<<endl;
  LogTrace(metname)<<"L2 Muon Reconstruction Started"<<endl;
  
  // Take the seeds container
  LogTrace(metname)<<"Taking the seeds: "<<theSeedCollectionLabel.label()<<endl;
  Handle<View<TrajectorySeed> > seeds; 
  event.getByToken(seedsToken,seeds);

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

