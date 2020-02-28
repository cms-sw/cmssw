// Author: Marco Rovere, marco.rovere@cern.ch
// Date: 05/2019
//
#include <memory>  // unique_ptr

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

class TICLLayerTileProducer : public edm::stream::EDProducer<> {
public:
  explicit TICLLayerTileProducer(const edm::ParameterSet &ps);
  ~TICLLayerTileProducer() override{};
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  hgcal::RecHitTools rhtools_;
};

TICLLayerTileProducer::TICLLayerTileProducer(const edm::ParameterSet &ps) {
  clusters_token_ = consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"));

  produces<TICLLayerTiles>();
}

void TICLLayerTileProducer::beginRun(edm::Run const &, edm::EventSetup const &es) { rhtools_.getEventSetup(es); }

void TICLLayerTileProducer::produce(edm::Event &evt, const edm::EventSetup &) {
  auto result = std::make_unique<TICLLayerTiles>();

  edm::Handle<std::vector<reco::CaloCluster>> cluster_h;
  evt.getByToken(clusters_token_, cluster_h);
  const auto &layerClusters = *cluster_h;
  int lcId = 0;
  for (auto const &lc : layerClusters) {
    const auto firstHitDetId = lc.hitsAndFractions()[0].first;
    int layer = rhtools_.getLayerWithOffset(firstHitDetId) +
                rhtools_.lastLayerFH() * ((rhtools_.zside(firstHitDetId) + 1) >> 1) - 1;
    assert(layer >= 0);
    result->fill(layer, lc.eta(), lc.phi(), lcId);
    LogDebug("TICLLayerTileProducer") << "Adding layerClusterId: " << lcId << " into bin [eta,phi]: [ "
                                      << (*result)[layer].etaBin(lc.eta()) << ", " << (*result)[layer].phiBin(lc.phi())
                                      << "] for layer: " << layer << std::endl;
    lcId++;
  }
  evt.put(std::move(result));
}

void TICLLayerTileProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  descriptions.add("ticlLayerTileProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLLayerTileProducer);
