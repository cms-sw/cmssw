// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch,marco.rovere@cern.ch
// Date: 09/2018

// user include files
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"

#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "PatternRecognitionbyCA.h"
#include "PatternRecognitionbyMultiClusters.h"

using namespace ticl;

class TrackstersProducer : public edm::stream::EDProducer<> {
public:
  TrackstersProducer(const edm::ParameterSet&);
  ~TrackstersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  edm::EDGetTokenT<std::vector<float>> filtered_layerclusters_mask_token_;
  edm::EDGetTokenT<std::vector<float>> original_layerclusters_mask_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> clustersTime_token_;
  edm::EDGetTokenT<TICLLayerTiles> layer_clusters_tiles_token_;
  edm::EDGetTokenT<std::vector<TICLSeedingRegion>> seeding_regions_token_;

  std::unique_ptr<PatternRecognitionAlgoBase> myAlgo_;
};
DEFINE_FWK_MODULE(TrackstersProducer);

TrackstersProducer::TrackstersProducer(const edm::ParameterSet& ps)
    : myAlgo_(std::make_unique<PatternRecognitionbyCA>(ps)) {
  clusters_token_ = consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"));
  filtered_layerclusters_mask_token_ = consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("filtered_mask"));
  original_layerclusters_mask_token_ = consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("original_mask"));
  clustersTime_token_ = consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("time_layerclusters"));
  layer_clusters_tiles_token_ = consumes<TICLLayerTiles>(ps.getParameter<edm::InputTag>("layer_clusters_tiles"));
  seeding_regions_token_ = consumes<std::vector<TICLSeedingRegion>>(ps.getParameter<edm::InputTag>("seeding_regions"));

  produces<std::vector<Trackster>>();
  produces<std::vector<float>>();  // Mask to be applied at the next iteration
}

void TrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("filtered_mask", edm::InputTag("filteredLayerClusters", "iterationLabelGoesHere"));
  desc.add<edm::InputTag>("original_mask", edm::InputTag("hgcalLayerClusters", "InitialLayerClustersMask"));
  desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hgcalLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("layer_clusters_tiles", edm::InputTag("ticlLayerTileProducer"));
  desc.add<edm::InputTag>("seeding_regions", edm::InputTag("ticlSeedingRegionProducer"));
  desc.add<int>("algo_verbosity", 0);
  desc.add<double>("min_cos_theta", 0.915);
  desc.add<double>("min_cos_pointing", -1.);
  desc.add<int>("missing_layers", 0);
  desc.add<int>("min_clusters_per_ntuplet", 10);
  desc.add<double>("max_delta_time", 0.09);
  desc.add<bool>("out_in_dfs", true);
  desc.add<int>("max_out_in_hops", 10);
  descriptions.add("trackstersProducer", desc);
}

void TrackstersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<Trackster>>();
  auto output_mask = std::make_unique<std::vector<float>>();

  edm::Handle<std::vector<reco::CaloCluster>> cluster_h;
  edm::Handle<std::vector<float>> filtered_layerclusters_mask_h;
  edm::Handle<std::vector<float>> original_layerclusters_mask_h;
  edm::Handle<edm::ValueMap<float>> time_clusters_h;
  edm::Handle<TICLLayerTiles> layer_clusters_tiles_h;
  edm::Handle<std::vector<TICLSeedingRegion>> seeding_regions_h;

  evt.getByToken(clusters_token_, cluster_h);
  evt.getByToken(filtered_layerclusters_mask_token_, filtered_layerclusters_mask_h);
  evt.getByToken(original_layerclusters_mask_token_, original_layerclusters_mask_h);
  evt.getByToken(clustersTime_token_, time_clusters_h);
  evt.getByToken(layer_clusters_tiles_token_, layer_clusters_tiles_h);
  evt.getByToken(seeding_regions_token_, seeding_regions_h);

  const auto& layerClusters = *cluster_h;
  const auto& inputClusterMask = *filtered_layerclusters_mask_h;
  const auto& layerClustersTimes = *time_clusters_h;
  const auto& layer_clusters_tiles = *layer_clusters_tiles_h;
  const auto& seeding_regions = *seeding_regions_h;
  const ticl::PatternRecognitionAlgoBase::Inputs input(
      evt, es, layerClusters, inputClusterMask, layerClustersTimes, layer_clusters_tiles, seeding_regions);
  myAlgo_->makeTracksters(input, *result);

  // Now update the global mask and put it into the event
  output_mask->reserve(original_layerclusters_mask_h->size());
  // Copy over the previous state
  std::copy(std::begin(*original_layerclusters_mask_h),
            std::end(*original_layerclusters_mask_h),
            std::back_inserter(*output_mask));
  // Mask the used elements, accordingly
  for (auto const& trackster : *result) {
    for (auto const v : trackster.vertices) {
      // TODO(rovere): for the moment we mask the layer cluster completely. In
      // the future, properly compute the fraction of usage.
      (*output_mask)[v] = 0.;
    }
  }

  evt.put(std::move(result));
  evt.put(std::move(output_mask));
}
