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
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"

#include "TrackstersPCA.h"

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
  const edm::EDGetTokenT<std::vector<TICLSeedingRegion>> seedingTrk_token_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
};

TrackstersMergeProducer::TrackstersMergeProducer(const edm::ParameterSet &ps) :
  trackstersem_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstersem"))),
  tracksterstrk_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("tracksterstrk"))),
  trackstershad_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstershad"))),
  seedingTrk_token_(consumes<std::vector<TICLSeedingRegion>>(ps.getParameter<edm::InputTag>("seedingTrk"))),
    clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))) {
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
  std::vector<bool> usedTrackstersEM;
  std::vector<bool> usedTrackstersTRK;
  std::vector<bool> usedTrackstersHAD;
  std::vector<bool> usedSeeds;

  edm::Handle<std::vector<reco::CaloCluster>> cluster_h;
  evt.getByToken(clusters_token_, cluster_h);
  const auto& layerClusters = *cluster_h;

  edm::Handle<std::vector<Trackster>> trackstersem_h;
  evt.getByToken(trackstersem_token_, trackstersem_h);
  const auto &trackstersEM = *trackstersem_h;
  usedTrackstersEM.resize(trackstersEM.size(), 0);

  edm::Handle<std::vector<Trackster>> tracksterstrk_h;
  evt.getByToken(tracksterstrk_token_, tracksterstrk_h);
  const auto &trackstersTRK = *tracksterstrk_h;
  usedTrackstersTRK.resize(trackstersTRK.size(), 0);

  edm::Handle<std::vector<Trackster>> trackstershad_h;
  evt.getByToken(trackstershad_token_, trackstershad_h);
  const auto &trackstersHAD = *trackstershad_h;
  usedTrackstersHAD.resize(trackstersHAD.size(), 0);

  edm::Handle<std::vector<TICLSeedingRegion>> seedingTrk_h;
  evt.getByToken(seedingTrk_token_, seedingTrk_h);
  const auto & seedingTrk = * seedingTrk_h;
  usedSeeds.resize(seedingTrk.size(), 0);

  int tracksterIteration = 0;
  fillTile(tracksterTile, trackstersEM, tracksterIteration++);
  fillTile(tracksterTile, trackstersTRK, tracksterIteration++);
  fillTile(tracksterTile, trackstersHAD, tracksterIteration++);

  auto seedId = 0;
  for (auto const & s : seedingTrk) {
    tracksterTile.fill(tracksterIteration, s.origin.eta(), s.origin.phi(), seedId++);
  }

  int tracksterTRK_idx = 0;
  for (auto const & t : trackstersTRK) {
    int bin = tracksterTile[3].globalBin(t.barycenter.eta(), t.barycenter.phi());
      std::cout << "TrackstersMergeProducer Tracking obj: " << t.barycenter
        << " regressed energy: " << t.regressed_energy
        << " raw_energy: " << t.raw_energy
        << std::endl;
    auto const & seeds = tracksterTile[3][bin];
    auto const & ems = tracksterTile[0][bin];
    auto const & original_seed = seedingTrk[t.seedIndex];
    std::cout << "Original seed: " << original_seed.origin
      << ", " << original_seed.directionAtOrigin.mag()
      << " trk/seed: " << t.raw_energy/original_seed.directionAtOrigin.mag()
      << " abs(alignemnt): " << std::abs(t.eigenvectors[0].Dot(original_seed.directionAtOrigin.unit()))
      << std::endl;
    if (t.raw_energy/original_seed.directionAtOrigin.mag() < 0.6) {
      std::cout << "Trying to recover energy..." << std::endl;
      auto tracksterEM_idx = 0;
      for (auto const & e : ems) {
        auto const & em = trackstersEM[e];
        auto cos_angle = std::abs(t.eigenvectors[0].Dot(em.eigenvectors[0]));
        if (cos_angle > 0.9945) {
          usedTrackstersTRK[tracksterTRK_idx] = 1;
          usedTrackstersEM[tracksterEM_idx] = 1;
          auto combined = t;
          std::copy(std::begin(em.vertices), std::end(em.vertices),
              std::back_inserter(combined.vertices));
          std::copy(std::begin(em.vertex_multiplicity), std::end(em.vertex_multiplicity),
              std::back_inserter(combined.vertex_multiplicity));
          std::cout << " linked to em obj: " << em.barycenter
            << " abs(alignemnt): " << std::abs(t.eigenvectors[0].Dot(em.eigenvectors[0]))
            << std::endl
            << " regressed energy: " << em.regressed_energy
            << " raw_energy: " << em.raw_energy
            << " cumulative: " << (t.raw_energy+em.raw_energy)
            << " (trk+em)/seed: " << (t.raw_energy+em.raw_energy)/original_seed.directionAtOrigin.mag()
            << std::endl;
          result->push_back(combined);
        }
        tracksterEM_idx++;
      }
    } else {
      result->push_back(t);
      usedTrackstersTRK[tracksterTRK_idx] = 1;
    }
    tracksterTRK_idx++;
  }

  tracksterTRK_idx = 0;
  for (auto const & t : trackstersTRK) {
    if (! usedTrackstersTRK[tracksterTRK_idx]) {
      result->push_back(t);
    }
    tracksterTRK_idx++;
  }

  auto tracksterEM_idx = 0;
  for (auto const & t : trackstersEM) {
    if (! usedTrackstersEM[tracksterEM_idx]) {
      result->push_back(t);
    }
    tracksterEM_idx++;
  }

  auto tracksterHAD_idx = 0;
  for (auto const & t : trackstersHAD) {
    if (! usedTrackstersHAD[tracksterHAD_idx]) {
      result->push_back(t);
    }
    tracksterHAD_idx++;
  }

  assignPCAtoTracksters(*result, layerClusters);

  evt.put(std::move(result));
}

void TrackstersMergeProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackstersem", edm::InputTag("trackstersEM"));
  desc.add<edm::InputTag>("tracksterstrk", edm::InputTag("trackstersTrk"));
  desc.add<edm::InputTag>("trackstershad", edm::InputTag("trackstersHAD"));
  desc.add<edm::InputTag>("seedingTrk", edm::InputTag("ticlSeedingTrk"));
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  descriptions.add("trackstersMergeProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackstersMergeProducer);
