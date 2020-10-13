#include "RecoMuon/CosmicMuonProducer/src/GlobalCosmicMuonProducer.h"

/**\class GlobalCosmicMuonProducer
 *
 *  reconstruct muons using dt,csc,rpc and tracker starting from cosmic muon
 *  tracks
 *
 * \author:  Chang Liu  - Purdue University <Chang.Liu@cern.ch>
**/

// system include files
#include <memory>
#include "FWCore/Framework/interface/ConsumesCollector.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "RecoMuon/CosmicMuonProducer/interface/GlobalCosmicMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

//
// constructors and destructor
//
GlobalCosmicMuonProducer::GlobalCosmicMuonProducer(const edm::ParameterSet& iConfig) {
  edm::ParameterSet tbpar = iConfig.getParameter<edm::ParameterSet>("TrajectoryBuilderParameters");
  theTrackCollectionToken = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("MuonCollectionLabel"));

  // service parameters
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  edm::ParameterSet trackLoaderParameters = iConfig.getParameter<edm::ParameterSet>("TrackLoaderParameters");

  // the services
  theService = std::make_unique<MuonServiceProxy>(serviceParameters, consumesCollector());
  edm::ConsumesCollector iC = consumesCollector();
  theTrackFinder = std::make_unique<MuonTrackFinder>(
      std::make_unique<GlobalCosmicMuonTrajectoryBuilder>(tbpar, theService.get(), iC),
      std::make_unique<MuonTrackLoader>(trackLoaderParameters, iC, theService.get()),
      iC);

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();

  produces<reco::MuonTrackLinksCollection>();
}

GlobalCosmicMuonProducer::~GlobalCosmicMuonProducer() {}

// ------------ method called to produce the data  ------------
void GlobalCosmicMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string metname = "Muon|RecoMuon|GlobalCosmicMuonProducer";
  LogTrace(metname) << "Global Cosmic Muon Reconstruction started";

  edm::Handle<reco::TrackCollection> cosMuons;
  iEvent.getByToken(theTrackCollectionToken, cosMuons);
  if (!cosMuons.isValid()) {
    LogTrace(metname) << "Muon Track collection is invalid!!!";
    return;
  }

  // Update the services
  theService->update(iSetup);

  // Reconstruct the tracks in the tracker+muon system
  LogTrace(metname) << "Track Reconstruction";

  std::vector<MuonTrajectoryBuilder::TrackCand> cosTrackCands;
  for (unsigned int position = 0; position != cosMuons->size(); ++position) {
    reco::TrackRef cosTrackRef(cosMuons, position);
    MuonTrajectoryBuilder::TrackCand cosCand = MuonTrajectoryBuilder::TrackCand((Trajectory*)nullptr, cosTrackRef);
    cosTrackCands.push_back(cosCand);
  }
  theTrackFinder->reconstruct(cosTrackCands, iEvent, iSetup);
  LogTrace(metname) << "Event loaded";
}
