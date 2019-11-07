// Author: Marco Rovere, marco.rovere@cern.ch
// Date: 11/2019
//
#include <memory>  // unique_ptr

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

using namespace ticl;

class TrackstersMergeProducer : public edm::stream::EDProducer<> {
public:
  explicit TrackstersMergeProducer(const edm::ParameterSet &ps);
  ~TrackstersMergeProducer() override{};
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void fillTile(TICLTracksterTiles &, const std::vector<Trackster> &, int);

  const edm::EDGetTokenT<std::vector<Trackster>> trackstersem_token_;
  const edm::EDGetTokenT<std::vector<Trackster>> tracksterstrk_token_;
  const edm::EDGetTokenT<std::vector<Trackster>> trackstershad_token_;
};

TrackstersMergeProducer::TrackstersMergeProducer(const edm::ParameterSet &ps) :
  trackstersem_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstersem"))),
  tracksterstrk_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("tracksterstrk"))),
  trackstershad_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstershad"))) {
  produces<std::vector<Trackster>>();
}

void TrackstersMergeProducer::fillTile(TICLTracksterTiles & tracksterTile,
    const std::vector<Trackster> & tracksters,
    int tracksterIteration) {
  int tracksterId = 0;
  for (auto const & t: tracksters) {
    tracksterTile.fill(tracksterIteration, t.barycenter.eta(), t.barycenter.phi(), tracksterId);
    LogDebug("TrackstersMergeProducer") << "Adding tracksterId: " << tracksterId << " into bin [eta,phi]: [ "
                                      << tracksterTile[tracksterIteration].etaBin(t.barycenter.eta())
                                      << ", " << tracksterTile[tracksterIteration].phiBin(t.barycenter.phi())
                                      << "] for iteration: " << tracksterIteration << std::endl;

    tracksterId++;
  }
}

void TrackstersMergeProducer::produce(edm::Event &evt, const edm::EventSetup &) {
  auto result = std::make_unique<std::vector<Trackster>>();

  TICLTracksterTiles tracksterTile;

  edm::Handle<std::vector<Trackster>> trackstersem_h;
  evt.getByToken(trackstersem_token_, trackstersem_h);
  const auto &trackstersEM = *trackstersem_h;

  edm::Handle<std::vector<Trackster>> tracksterstrk_h;
  evt.getByToken(tracksterstrk_token_, tracksterstrk_h);
  const auto &trackstersTRK = *tracksterstrk_h;

  edm::Handle<std::vector<Trackster>> trackstershad_h;
  evt.getByToken(trackstershad_token_, trackstershad_h);
  const auto &trackstersHAD = *trackstershad_h;

  int tracksterIteration = 0;
  fillTile(tracksterTile, trackstersEM, tracksterIteration++);
  fillTile(tracksterTile, trackstersTRK, tracksterIteration++);
  fillTile(tracksterTile, trackstersHAD, tracksterIteration++);
  evt.put(std::move(result));
}

void TrackstersMergeProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackstersem", edm::InputTag("trackstersEM"));
  desc.add<edm::InputTag>("tracksterstrk", edm::InputTag("trackstersTrk"));
  desc.add<edm::InputTag>("trackstershad", edm::InputTag("trackstersHAD"));
  descriptions.add("TrackstersMergeProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackstersMergeProducer);
