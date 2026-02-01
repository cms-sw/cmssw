// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      Run3ScoutingTrackToRecoTrackProducer
//
/**\class Run3ScoutingTrackToRecoTrackProducer Run3ScoutingTrackToRecoTrackProducer.cc PhysicsTools/PatFromScouting/plugins/Run3ScoutingTrackToRecoTrackProducer.cc

 Description: Converts Run3ScoutingTrack to reco::Track

 Implementation:
     Uses pat::makeRecoTrack helper, stores hit pattern info as ValueMaps
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Thu, 05 Dec 2024 15:27:09 GMT
//
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/PatCandidates/interface/ScoutingDataHandling.h"

class Run3ScoutingTrackToRecoTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit Run3ScoutingTrackToRecoTrackProducer(const edm::ParameterSet&);
  ~Run3ScoutingTrackToRecoTrackProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingTrackCollection> trackToken_;

  std::vector<int> nValidPixelHits_;
  std::vector<int> nValidStripHits_;
  std::vector<int> nTrackerLayers_;
  std::vector<int> vertexIndex_;
};

Run3ScoutingTrackToRecoTrackProducer::Run3ScoutingTrackToRecoTrackProducer(const edm::ParameterSet& iConfig)
    : trackToken_(consumes<Run3ScoutingTrackCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<reco::TrackCollection>();
  produces<edm::ValueMap<int>>("nValidPixelHits");
  produces<edm::ValueMap<int>>("nValidStripHits");
  produces<edm::ValueMap<int>>("nTrackerLayersWithMeasurement");
  produces<edm::ValueMap<int>>("vertexIndex");
}

void Run3ScoutingTrackToRecoTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto recoTracks = std::make_unique<reco::TrackCollection>();

  nValidPixelHits_.clear();
  nValidStripHits_.clear();
  nTrackerLayers_.clear();
  vertexIndex_.clear();

  const auto& scoutingTracks = iEvent.get(trackToken_);

  for (const auto& sTrack : scoutingTracks) {
    reco::Track recoTrack = pat::makeRecoTrack(sTrack);
    recoTracks->push_back(recoTrack);

    nValidPixelHits_.push_back(sTrack.tk_nValidPixelHits());
    nValidStripHits_.push_back(sTrack.tk_nValidStripHits());
    nTrackerLayers_.push_back(sTrack.tk_nTrackerLayersWithMeasurement());
    vertexIndex_.push_back(sTrack.tk_vtxInd());
  }

  edm::OrphanHandle<reco::TrackCollection> tracksHandle = iEvent.put(std::move(recoTracks));

  auto nValidPixelHitsMap = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler fillerPixel(*nValidPixelHitsMap);
  fillerPixel.insert(tracksHandle, nValidPixelHits_.begin(), nValidPixelHits_.end());
  fillerPixel.fill();
  iEvent.put(std::move(nValidPixelHitsMap), "nValidPixelHits");

  auto nValidStripHitsMap = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler fillerStrip(*nValidStripHitsMap);
  fillerStrip.insert(tracksHandle, nValidStripHits_.begin(), nValidStripHits_.end());
  fillerStrip.fill();
  iEvent.put(std::move(nValidStripHitsMap), "nValidStripHits");

  auto nTrackerLayersMap = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler fillerLayers(*nTrackerLayersMap);
  fillerLayers.insert(tracksHandle, nTrackerLayers_.begin(), nTrackerLayers_.end());
  fillerLayers.fill();
  iEvent.put(std::move(nTrackerLayersMap), "nTrackerLayersWithMeasurement");

  auto vertexIndexMap = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler fillerVtx(*vertexIndexMap);
  fillerVtx.insert(tracksHandle, vertexIndex_.begin(), vertexIndex_.end());
  fillerVtx.fill();
  iEvent.put(std::move(vertexIndexMap), "vertexIndex");
}

void Run3ScoutingTrackToRecoTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingTrackPacker"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Run3ScoutingTrackToRecoTrackProducer);
