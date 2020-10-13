/**  \class StandAloneMuonProducer
 * 
 *   StandAlone muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *
 *   \author  R.Bellan - INFN TO
 */

// Framework
#include "FWCore/Framework/interface/ConsumesCollector.h"
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

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include <string>

using namespace edm;
using namespace std;

/// constructor with config
StandAloneMuonProducer::StandAloneMuonProducer(const ParameterSet& parameterSet) {
  LogTrace("Muon|RecoMuon|StandAloneMuonProducer") << "constructor called" << endl;

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("STATrajBuilderParameters");

  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");
  edm::ConsumesCollector iC = consumesCollector();

  // the services
  theService = std::make_unique<MuonServiceProxy>(serviceParameters, consumesCollector());

  auto trackLoader = std::make_unique<MuonTrackLoader>(trackLoaderParameters, iC, theService.get());
  std::unique_ptr<MuonTrajectoryBuilder> trajectoryBuilder;
  // instantiate the concrete trajectory builder in the Track Finder
  string typeOfBuilder = parameterSet.getParameter<string>("MuonTrajectoryBuilder");
  if (typeOfBuilder == "StandAloneMuonTrajectoryBuilder")
    trajectoryBuilder =
        std::make_unique<StandAloneMuonTrajectoryBuilder>(trajectoryBuilderParameters, theService.get(), iC);
  else if (typeOfBuilder == "DirectMuonTrajectoryBuilder")
    trajectoryBuilder = std::make_unique<DirectMuonTrajectoryBuilder>(trajectoryBuilderParameters, theService.get());
  else if (typeOfBuilder == "Exhaustive")
    trajectoryBuilder =
        std::make_unique<ExhaustiveMuonTrajectoryBuilder>(trajectoryBuilderParameters, theService.get(), iC);
  else {
    LogWarning("Muon|RecoMuon|StandAloneMuonProducer")
        << "No Trajectory builder associated with " << typeOfBuilder
        << ". Falling down to the default (StandAloneMuonTrajectoryBuilder)";
    trajectoryBuilder =
        std::make_unique<StandAloneMuonTrajectoryBuilder>(trajectoryBuilderParameters, theService.get(), iC);
  }
  theTrackFinder = std::make_unique<MuonTrackFinder>(std::move(trajectoryBuilder), std::move(trackLoader), iC);

  setAlias(parameterSet.getParameter<std::string>("@module_label"));

  produces<reco::TrackCollection>().setBranchAlias(theAlias + "Tracks");
  produces<reco::TrackCollection>("UpdatedAtVtx").setBranchAlias(theAlias + "UpdatedAtVtxTracks");
  produces<TrackingRecHitCollection>().setBranchAlias(theAlias + "RecHits");
  produces<reco::TrackExtraCollection>().setBranchAlias(theAlias + "TrackExtras");
  produces<reco::TrackToTrackMap>().setBranchAlias(theAlias + "TrackToTrackMap");

  produces<std::vector<Trajectory> >().setBranchAlias(theAlias + "Trajectories");
  produces<TrajTrackAssociationCollection>().setBranchAlias(theAlias + "TrajToTrackMap");

  seedToken = consumes<edm::View<TrajectorySeed> >(theSeedCollectionLabel);
}

/// destructor
StandAloneMuonProducer::~StandAloneMuonProducer() {
  LogTrace("Muon|RecoMuon|StandAloneMuonProducer") << "StandAloneMuonProducer destructor called" << endl;
}

/// reconstruct muons
void StandAloneMuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const std::string metname = "Muon|RecoMuon|StandAloneMuonProducer";

  LogTrace(metname) << endl << endl << endl;
  LogTrace(metname) << "Stand Alone Muon Reconstruction Started" << endl;

  // Take the seeds container
  LogTrace(metname) << "Taking the seeds: " << theSeedCollectionLabel.label() << endl;
  Handle<View<TrajectorySeed> > seeds;
  event.getByToken(seedToken, seeds);

  // Update the services
  theService->update(eventSetup);

  // Reconstruct
  LogTrace(metname) << "Track Reconstruction" << endl;
  theTrackFinder->reconstruct(seeds, event, eventSetup);

  LogTrace(metname) << "Event loaded"
                    << "================================" << endl
                    << endl;
}
