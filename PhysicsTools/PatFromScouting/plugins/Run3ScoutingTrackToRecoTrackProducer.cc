// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      Run3ScoutingTrackToRecoTrackProducer
//
/**\class Run3ScoutingTrackToRecoTrackProducer Run3ScoutingTrackToRecoTrackProducer.cc PhysicsTools/PatFromScouting/plugins/Run3ScoutingTrackToRecoTrackProducer.cc

 Description: Converts Run3ScoutingTrack to reco::Track with populated hit pattern

*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Thu, 05 Dec 2024 15:27:09 GMT
//
//

#include <memory>
#include <algorithm>

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
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

class Run3ScoutingTrackToRecoTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit Run3ScoutingTrackToRecoTrackProducer(const edm::ParameterSet&);
  ~Run3ScoutingTrackToRecoTrackProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingTrackCollection> trackToken_;

  std::vector<int> vertexIndex_;
};

Run3ScoutingTrackToRecoTrackProducer::Run3ScoutingTrackToRecoTrackProducer(const edm::ParameterSet& iConfig)
    : trackToken_(consumes<Run3ScoutingTrackCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<reco::TrackCollection>();
  produces<edm::ValueMap<int>>("vertexIndex");
}

void Run3ScoutingTrackToRecoTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto recoTracks = std::make_unique<reco::TrackCollection>();

  vertexIndex_.clear();

  const auto& scoutingTracks = iEvent.get(trackToken_);

  for (const auto& sTrack : scoutingTracks) {
    reco::Track recoTrack = pat::makeRecoTrack(sTrack);

    // Populate hit pattern from scouting hit counts.
    // Prefer pixLayers = min(nPixHits, 4) but adjust to guarantee:
    //   pixLayers + stripLayers = totalLayers, pixLayers <= nPixHits, stripLayers <= nStripHits
    int nPixHits = sTrack.tk_nValidPixelHits();
    int nStripHits = sTrack.tk_nValidStripHits();
    int totalLayers = sTrack.tk_nTrackerLayersWithMeasurement();
    int lower = std::max(0, totalLayers - nStripHits);
    int upper = nStripHits > 0 ? std::min(nPixHits, totalLayers - 1) : std::min(nPixHits, totalLayers);
    int pixLayers = std::clamp(std::min(4, totalLayers), lower, upper);
    int stripLayers = totalLayers - pixLayers;

    // Pixel: BPIX(1-4), FPIX(1-3), extras on BPIX layer 2
    for (int i = 0; i < pixLayers; ++i) {
      if (i < 4)
        recoTrack.appendTrackerHitPattern(PixelSubdetector::PixelBarrel, i + 1, 0, TrackingRecHit::valid);
      else
        recoTrack.appendTrackerHitPattern(PixelSubdetector::PixelEndcap, i - 3, 0, TrackingRecHit::valid);
    }
    for (int i = pixLayers; i < nPixHits; ++i)
      recoTrack.appendTrackerHitPattern(PixelSubdetector::PixelBarrel, 2, 0, TrackingRecHit::valid);

    // Strip: TIB(1-4), TID(1-3), TOB(1-6), TEC(1-9), extras on TIB 1
    for (int i = 0; i < stripLayers; ++i) {
      if (i < 4)
        recoTrack.appendTrackerHitPattern(StripSubdetector::TIB, i + 1, 1, TrackingRecHit::valid);
      else if (i < 7)
        recoTrack.appendTrackerHitPattern(StripSubdetector::TID, i - 3, 1, TrackingRecHit::valid);
      else if (i < 13)
        recoTrack.appendTrackerHitPattern(StripSubdetector::TOB, i - 6, 1, TrackingRecHit::valid);
      else
        recoTrack.appendTrackerHitPattern(StripSubdetector::TEC, i - 12, 1, TrackingRecHit::valid);
    }
    for (int i = stripLayers; i < nStripHits; ++i)
      recoTrack.appendTrackerHitPattern(StripSubdetector::TIB, 1, 1, TrackingRecHit::valid);

    recoTracks->push_back(recoTrack);

    vertexIndex_.push_back(sTrack.tk_vtxInd());
  }

  edm::OrphanHandle<reco::TrackCollection> tracksHandle = iEvent.put(std::move(recoTracks));

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
