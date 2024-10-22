// Author: Marco Rovere, marco.rovere@cern.ch
// Date: 05/2019
//
#include <memory>  // unique_ptr

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

class TICLLayerTileProducer : public edm::stream::EDProducer<> {
public:
  explicit TICLLayerTileProducer(const edm::ParameterSet &ps);
  ~TICLLayerTileProducer() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_barrel_token_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_HFNose_token_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  hgcal::RecHitTools rhtools_;
  std::string detector_;
  bool doNose_;
  bool doBarrel_;
};

TICLLayerTileProducer::TICLLayerTileProducer(const edm::ParameterSet &ps)
    : detector_(ps.getParameter<std::string>("detector")) {
  geometry_token_ = esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>();

  doNose_ = (detector_ == "HFNose");
  doBarrel_ = (detector_ == "Barrel");

  if (doNose_) {
    clusters_HFNose_token_ =
        consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_HFNose_clusters"));
    produces<TICLLayerTilesHFNose>();
  } else if (doBarrel_) {
    clusters_barrel_token_ = 
        consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("barrel_layer_clusters"));
    produces<TICLLayerTilesBarrel>("ticlLayerTilesBarrel");
  } else {
    clusters_token_ = consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"));
    produces<TICLLayerTiles>();
  }
}

void TICLLayerTileProducer::beginRun(edm::Run const &, edm::EventSetup const &es) {
  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);
}

void TICLLayerTileProducer::produce(edm::Event &evt, const edm::EventSetup &) {
  std::unique_ptr<TICLLayerTilesHFNose> resultHFNose;
  std::unique_ptr<TICLLayerTiles> result;
  std::unique_ptr<TICLLayerTilesBarrel> resultBarrel;
  if (doNose_) {
    resultHFNose = std::make_unique<TICLLayerTilesHFNose>();
  } else if (doBarrel_) {
    resultBarrel = std::make_unique<TICLLayerTilesBarrel>();
  } else {
    result = std::make_unique<TICLLayerTiles>();
  }

  edm::Handle<std::vector<reco::CaloCluster>> cluster_h;
  if (doNose_)
    evt.getByToken(clusters_HFNose_token_, cluster_h);
  else if (doBarrel_)
    evt.getByToken(clusters_barrel_token_, cluster_h);
  else
    evt.getByToken(clusters_token_, cluster_h);

  const auto &layerClusters = *cluster_h;
  int lcId = 0;
  for (auto const &lc : layerClusters) {
    const auto firstHitDetId = lc.hitsAndFractions()[0].first;
    int layer = rhtools_.getLayerWithOffset(firstHitDetId);
    if (!doBarrel_)
    layer += rhtools_.getLayerWithOffset(firstHitDetId) +
             rhtools_.lastLayer(doNose_) * ((rhtools_.zside(firstHitDetId) + 1) >> 1) - 1;

    assert(layer >= 0);

    if (doNose_) {
      resultHFNose->fill(layer, lc.eta(), lc.phi(), lcId);
    } else if (doBarrel_) {
      resultBarrel->fill(layer, lc.eta(), lc.phi(), lcId);
    } else {
      result->fill(layer, lc.eta(), lc.phi(), lcId);
      LogDebug("TICLLayerTileProducer") << "Adding layerClusterId: " << lcId << " into bin [eta,phi]: [ "
                                        << (*result)[layer].etaBin(lc.eta()) << ", "
                                        << (*result)[layer].phiBin(lc.phi()) << "] for layer: " << layer << std::endl;
    }
    lcId++;
  }
  if (doNose_)
    evt.put(std::move(resultHFNose));
  else if (doBarrel_)
    evt.put(std::move(resultBarrel), "ticlLayerTilesBarrel");
  else
    evt.put(std::move(result));
}

void TICLLayerTileProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("barrel_layer_clusters", edm::InputTag("barrelLayerClusters"));
  desc.add<edm::InputTag>("layer_HFNose_clusters", edm::InputTag("hgcalLayerClustersHFNose"));
  descriptions.add("ticlLayerTileProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLLayerTileProducer);
