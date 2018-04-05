#include "RecoMuon/CosmicMuonProducer/src/CosmicMuonProducer.h"

/**\class CosmicMuonProducer
 *
 * Description: CosmicMuonProducer for muons from cosmic rays
 *
 * Implementation:
 *
 * Original Author:  Chang Liu
 *        Created:  Tue Jun 13 02:46:17 CEST 2006
**/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

using namespace edm;

//
// constructors and destructor
//
CosmicMuonProducer::CosmicMuonProducer(const ParameterSet& iConfig)
{
  ParameterSet tbpar = iConfig.getParameter<ParameterSet>("TrajectoryBuilderParameters");

  theSeedCollectionToken =consumes<edm::View<TrajectorySeed> >( iConfig.getParameter<std::string>("MuonSeedCollectionLabel"));

  // service parameters
  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  
  // TrackLoader parameters
  ParameterSet trackLoaderParameters = iConfig.getParameter<ParameterSet>("TrackLoaderParameters");
  
  // the services

  edm::ConsumesCollector iC = consumesCollector();
  
  theService = new MuonServiceProxy(serviceParameters);
  theTrackFinder = new MuonTrackFinder(new CosmicMuonTrajectoryBuilder(tbpar,theService,iC),
				       new MuonTrackLoader(trackLoaderParameters,iC, theService));

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();

}


CosmicMuonProducer::~CosmicMuonProducer()
{
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;
}


// ------------ method called to produce the data  ------------
void
CosmicMuonProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  LogInfo("CosmicMuonProducer") << "Analyzing event number: " << iEvent.id();

  Handle<View<TrajectorySeed> > seeds; 
  iEvent.getByToken(theSeedCollectionToken,seeds);
  
  // Update the services
  theService->update(iSetup);
  theTrackFinder->reconstruct(seeds,iEvent,iSetup);
}
