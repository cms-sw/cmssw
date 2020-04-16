/**  \class TevMuonProducer
 * 
 *   TeV muon reconstructor:
 *
 *
 *
 *   \author  Piotr Traczyk (SINS Warsaw)
 */

// Framework
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoMuon/GlobalMuonProducer/src/TevMuonProducer.h"

// TrackFinder and specific GLB Trajectory Builder
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonRefitter.h"

using namespace edm;
using namespace std;
using namespace reco;

//
// constructor with config
//
TevMuonProducer::TevMuonProducer(const ParameterSet& parameterSet) {
  LogDebug("Muon|RecoMuon|TevMuonProducer") << "constructor called" << endl;

  // GLB Muon Collection Label
  theGLBCollectionLabel = parameterSet.getParameter<InputTag>("MuonCollectionLabel");
  glbMuonsToken = consumes<reco::TrackCollection>(theGLBCollectionLabel);
  glbMuonsTrajToken = consumes<std::vector<Trajectory> >(theGLBCollectionLabel.label());

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // the services
  theService = std::make_unique<MuonServiceProxy>(serviceParameters, consumesCollector());
  edm::ConsumesCollector iC = consumesCollector();

  // TrackRefitter parameters
  ParameterSet refitterParameters = parameterSet.getParameter<ParameterSet>("RefitterParameters");
  theRefitter = std::make_unique<GlobalMuonRefitter>(refitterParameters, theService.get(), iC);

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");
  theTrackLoader = std::make_unique<MuonTrackLoader>(trackLoaderParameters, iC, theService.get());

  theRefits = parameterSet.getParameter<std::vector<std::string> >("Refits");
  theRefitIndex = parameterSet.getParameter<std::vector<int> >("RefitIndex");

  for (unsigned int ww = 0; ww < theRefits.size(); ww++) {
    LogDebug("Muon|RecoMuon|TevMuonProducer") << "Refit " << theRefits[ww];
    produces<reco::TrackCollection>(theRefits[ww]);
    produces<TrackingRecHitCollection>(theRefits[ww]);
    produces<reco::TrackExtraCollection>(theRefits[ww]);
    produces<vector<Trajectory> >(theRefits[ww]);
    produces<TrajTrackAssociationCollection>(theRefits[ww]);
    produces<reco::TrackToTrackMap>(theRefits[ww]);
  }
  produces<DYTestimators>("dytInfo");
}

//
// destructor
//
TevMuonProducer::~TevMuonProducer() { LogTrace("Muon|RecoMuon|TevMuonProducer") << "destructor called" << endl; }

//
// reconstruct muons
//
void TevMuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const string metname = "Muon|RecoMuon|TevMuonProducer";
  LogTrace(metname) << endl << endl;
  LogTrace(metname) << "TeV Muon Reconstruction started" << endl;

  // Update the services
  theService->update(eventSetup);

  theRefitter->setEvent(event);

  theRefitter->setServices(theService->eventSetup());

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  eventSetup.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology* tTopo = tTopoHand.product();

  // Take the GLB muon container(s)
  Handle<reco::TrackCollection> glbMuons;
  event.getByToken(glbMuonsToken, glbMuons);

  auto dytInfo = std::make_unique<DYTestimators>();
  DYTestimators::Filler filler(*dytInfo);
  size_t GLBmuonSize = glbMuons->size();
  vector<DYTInfo> dytTmp(GLBmuonSize);

  Handle<vector<Trajectory> > glbMuonsTraj;

  LogTrace(metname) << "Taking " << glbMuons->size() << " Global Muons " << theGLBCollectionLabel << endl;

  vector<MuonTrajectoryBuilder::TrackCand> glbTrackCands;

  event.getByToken(glbMuonsTrajToken, glbMuonsTraj);

  const reco::TrackCollection* glbTracks = glbMuons.product();

  for (unsigned int ww = 0; ww < theRefits.size(); ww++) {
    LogDebug(metname) << "TeVRefit for Refit: " << theRefitIndex[ww];
    std::vector<std::pair<Trajectory*, reco::TrackRef> > miniMap;
    MuonTrackLoader::TrajectoryContainer trajectories;
    reco::TrackRef::key_type trackIndex = 0;
    int glbCounter = 0;
    for (reco::TrackCollection::const_iterator track = glbTracks->begin(); track != glbTracks->end();
         track++, ++trackIndex) {
      reco::TrackRef glbRef(glbMuons, trackIndex);

      vector<Trajectory> refitted = theRefitter->refit(*track, theRefitIndex[ww], tTopo);

      if (theRefits[ww] == "dyt")
        dytTmp[glbCounter] = *theRefitter->getDYTInfo();
      glbCounter++;

      if (!refitted.empty()) {
        auto refit = std::make_unique<Trajectory>(refitted.front());
        LogDebug(metname) << "TeVTrackLoader for Refit: " << theRefits[ww];
        std::pair<Trajectory*, reco::TrackRef> thisPair(refit.get(), glbRef);
        miniMap.push_back(thisPair);
        trajectories.push_back(std::move(refit));
      }
    }
    theTrackLoader->loadTracks(trajectories, event, miniMap, glbMuons, *tTopo, theRefits[ww]);
  }

  filler.insert(glbMuons, dytTmp.begin(), dytTmp.end());
  filler.fill();
  event.put(std::move(dytInfo), "dytInfo");

  LogTrace(metname) << "Done." << endl;
}
