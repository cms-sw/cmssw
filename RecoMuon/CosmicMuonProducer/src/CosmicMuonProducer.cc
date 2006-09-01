#include "RecoMuon/CosmicMuonProducer/src/CosmicMuonProducer.h"

/**\class CosmicMuonProducer
 *
 * Description: CosmicMuonProducer for muons from cosmic rays
 *
 * Implementation:
 *
 * $Date: 2006/09/01 13:11:53 $
 * $Revision: 1.8 $
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

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

//
// constructors and destructor
//
CosmicMuonProducer::CosmicMuonProducer(const edm::ParameterSet& iConfig)
{

  edm::ParameterSet tbpar = iConfig.getParameter<edm::ParameterSet>("TrajectoryBuilderParameters");
  theSeedCollectionLabel = iConfig.getUntrackedParameter<std::string>("MuonSeedCollectionLabel");

  // service parameters
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  // the propagator name for the track loader
  std::string trackLoaderPropagatorName = iConfig.getParameter<std::string>("TrackLoaderPropagatorName");
  
  theTrackFinder = new MuonTrackFinder(new CosmicMuonTrajectoryBuilder(tbpar,theService),
				       new MuonTrackLoader(trackLoaderPropagatorName,theService));

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();

}


CosmicMuonProducer::~CosmicMuonProducer()
{
  if (theTrackFinder) delete theTrackFinder;
}


// ------------ method called to produce the data  ------------
void
CosmicMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::LogInfo("CosmicMuonProducer") << "Analyzing event number: " << iEvent.id();

  edm::Handle<TrajectorySeedCollection> seeds; 
  iEvent.getByLabel(theSeedCollectionLabel,seeds);

  // Update the services
  theService->update(iSetup);
  theTrackFinder->reconstruct(seeds,iEvent);

}

//define this as a plug-in
DEFINE_FWK_MODULE(CosmicMuonProducer)
