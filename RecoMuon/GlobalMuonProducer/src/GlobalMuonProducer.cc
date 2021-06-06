/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a standalone reonstructed muon.
 *
 *
 *   \author  R.Bellan - INFN TO
 */

// Framework
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonProducer.h"

// TrackFinder and specific GLB Trajectory Builder
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

using namespace edm;
using namespace std;

//
// constructor with config
//
GlobalMuonProducer::GlobalMuonProducer(const ParameterSet& parameterSet) {
  LogTrace("Muon|RecoMuon|GlobalMuonProducer") << "constructor called" << endl;

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("GLBTrajBuilderParameters");
  InputTag trackCollectionTag = parameterSet.getParameter<InputTag>("TrackerCollectionLabel");
  trajectoryBuilderParameters.addParameter<InputTag>("TrackerCollectionLabel", trackCollectionTag);

  // STA Muon Collection Label
  theSTACollectionLabel = parameterSet.getParameter<InputTag>("MuonCollectionLabel");
  staMuonsToken = consumes<reco::TrackCollection>(parameterSet.getParameter<InputTag>("MuonCollectionLabel"));
  staMuonsTrajToken =
      consumes<std::vector<Trajectory> >(parameterSet.getParameter<InputTag>("MuonCollectionLabel").label());
  staAssoMapToken =
      consumes<TrajTrackAssociationCollection>(parameterSet.getParameter<InputTag>("MuonCollectionLabel").label());
  updatedStaAssoMapToken =
      consumes<reco::TrackToTrackMap>(parameterSet.getParameter<InputTag>("MuonCollectionLabel").label());

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");

  // the services
  theService = new MuonServiceProxy(serviceParameters, consumesCollector());

  // instantiate the concrete trajectory builder in the Track Finder
  edm::ConsumesCollector iC = consumesCollector();
  auto mtl = std::make_unique<MuonTrackLoader>(trackLoaderParameters, iC, theService);
  auto gmtb = std::make_unique<GlobalMuonTrajectoryBuilder>(trajectoryBuilderParameters, theService, iC);

  theTrackFinder = std::make_unique<MuonTrackFinder>(std::move(gmtb), std::move(mtl), iC);

  setAlias(parameterSet.getParameter<std::string>("@module_label"));
  produces<reco::TrackCollection>().setBranchAlias(theAlias + "Tracks");
  produces<TrackingRecHitCollection>().setBranchAlias(theAlias + "RecHits");
  produces<reco::TrackExtraCollection>().setBranchAlias(theAlias + "TrackExtras");
  produces<vector<Trajectory> >().setBranchAlias(theAlias + "Trajectories");
  produces<TrajTrackAssociationCollection>().setBranchAlias(theAlias + "TrajTrackMap");
  produces<reco::MuonTrackLinksCollection>().setBranchAlias(theAlias + "s");
}

//
// destructor
//
GlobalMuonProducer::~GlobalMuonProducer() {
  LogTrace("Muon|RecoMuon|GlobalMuonProducer") << "destructor called" << endl;
  if (theService)
    delete theService;
}

//
// reconstruct muons
//
void GlobalMuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const string metname = "Muon|RecoMuon|GlobalMuonProducer";
  LogTrace(metname) << endl << endl << endl;
  LogTrace(metname) << "Global Muon Reconstruction started" << endl;

  // Update the services
  theService->update(eventSetup);

  // Take the STA muon container(s)
  Handle<reco::TrackCollection> staMuons;
  event.getByToken(staMuonsToken, staMuons);

  Handle<vector<Trajectory> > staMuonsTraj;

  LogTrace(metname) << "Taking " << staMuons->size() << " Stand Alone Muons " << endl;

  vector<MuonTrajectoryBuilder::TrackCand> staTrackCands;

  edm::Handle<TrajTrackAssociationCollection> staAssoMap;

  edm::Handle<reco::TrackToTrackMap> updatedStaAssoMap;

  if (event.getByToken(staMuonsTrajToken, staMuonsTraj) && event.getByToken(staAssoMapToken, staAssoMap) &&
      event.getByToken(updatedStaAssoMapToken, updatedStaAssoMap)) {
    for (TrajTrackAssociationCollection::const_iterator it = staAssoMap->begin(); it != staAssoMap->end(); ++it) {
      const Ref<vector<Trajectory> > traj = it->key;
      const reco::TrackRef tkRegular = it->val;
      reco::TrackRef tkUpdated;
      reco::TrackToTrackMap::const_iterator iEnd;
      reco::TrackToTrackMap::const_iterator iii;
      if (theSTACollectionLabel.instance() == "UpdatedAtVtx") {
        iEnd = updatedStaAssoMap->end();
        iii = updatedStaAssoMap->find(it->val);
        if (iii != iEnd)
          tkUpdated = (*updatedStaAssoMap)[it->val];
      }

      int etaFlip1 =
          ((tkUpdated.isNonnull() && tkRegular.isNonnull()) && ((tkUpdated->eta() * tkRegular->eta()) < 0)) ? -1 : 1;

      const reco::TrackRef tk = (tkUpdated.isNonnull() && etaFlip1 == 1) ? tkUpdated : tkRegular;

      MuonTrajectoryBuilder::TrackCand tkCand = MuonTrajectoryBuilder::TrackCand((Trajectory*)nullptr, tk);
      if (traj->isValid())
        tkCand.first = &*traj;
      staTrackCands.push_back(tkCand);
    }
  } else {
    for (unsigned int position = 0; position != staMuons->size(); ++position) {
      reco::TrackRef staTrackRef(staMuons, position);
      MuonTrajectoryBuilder::TrackCand staCand = MuonTrajectoryBuilder::TrackCand((Trajectory*)nullptr, staTrackRef);
      staTrackCands.push_back(staCand);
    }
  }

  theTrackFinder->reconstruct(staTrackCands, event, eventSetup);

  LogTrace(metname) << "Event loaded"
                    << "================================" << endl
                    << endl;
}
