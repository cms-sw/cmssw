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

#include "RecoHGCal/TICL/plugins/PatternRecognitionAlgoBase.h"
#include "RecoHGCal/TICL/plugins/GlobalCache.h"
#include "PatternRecognitionbyCA.h"
#include "PatternRecognitionbyMultiClusters.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

using namespace ticl;

class TrackstersProducer : public edm::stream::EDProducer<edm::GlobalCache<TrackstersCache>> {
public:
  explicit TrackstersProducer(const edm::ParameterSet&, const TrackstersCache*);
  ~TrackstersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

  // static methods for handling the global cache
  static std::unique_ptr<TrackstersCache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(TrackstersCache*);

private:
  std::string detector_;
  bool doNose_;
  std::unique_ptr<PatternRecognitionAlgoBaseT<TICLLayerTiles>> myAlgo_;
  std::unique_ptr<PatternRecognitionAlgoBaseT<TICLLayerTilesHFNose>> myAlgoHFNose_;

  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<std::vector<float>> filtered_layerclusters_mask_token_;
  const edm::EDGetTokenT<std::vector<float>> original_layerclusters_mask_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  edm::EDGetTokenT<TICLLayerTiles> layer_clusters_tiles_token_;
  edm::EDGetTokenT<TICLLayerTilesHFNose> layer_clusters_tiles_hfnose_token_;
  const edm::EDGetTokenT<std::vector<TICLSeedingRegion>> seeding_regions_token_;
  const std::string itername_;
};
DEFINE_FWK_MODULE(TrackstersProducer);

std::unique_ptr<TrackstersCache> TrackstersProducer::initializeGlobalCache(const edm::ParameterSet& params) {
  // this method is supposed to create, initialize and return a TrackstersCache instance
  std::unique_ptr<TrackstersCache> cache = std::make_unique<TrackstersCache>(params);

  // load the graph def and save it
  std::string graphPath = params.getParameter<std::string>("eid_graph_path");
  if (!graphPath.empty()) {
    graphPath = edm::FileInPath(graphPath).fullPath();
    cache->eidGraphDef = tensorflow::loadGraphDef(graphPath);
  }

  return cache;
}

void TrackstersProducer::globalEndJob(TrackstersCache* cache) {
  delete cache->eidGraphDef;
  cache->eidGraphDef = nullptr;
}

TrackstersProducer::TrackstersProducer(const edm::ParameterSet& ps, const TrackstersCache* cache)
    : detector_(ps.getParameter<std::string>("detector")),
      doNose_(detector_ == "HFNose"),
      myAlgo_(std::make_unique<PatternRecognitionbyCA<TICLLayerTiles>>(ps, cache)),
      myAlgoHFNose_(std::make_unique<PatternRecognitionbyCA<TICLLayerTilesHFNose>>(ps, cache)),
      clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      filtered_layerclusters_mask_token_(consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("filtered_mask"))),
      original_layerclusters_mask_token_(consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("original_mask"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("time_layerclusters"))),
      seeding_regions_token_(
          consumes<std::vector<TICLSeedingRegion>>(ps.getParameter<edm::InputTag>("seeding_regions"))),
      itername_(ps.getParameter<std::string>("itername")) {
  if (doNose_) {
    layer_clusters_tiles_hfnose_token_ =
        consumes<TICLLayerTilesHFNose>(ps.getParameter<edm::InputTag>("layer_clusters_hfnose_tiles"));
  } else {
    layer_clusters_tiles_token_ = consumes<TICLLayerTiles>(ps.getParameter<edm::InputTag>("layer_clusters_tiles"));
  }
  produces<std::vector<Trackster>>();
  produces<std::vector<float>>();  // Mask to be applied at the next iteration
}

void TrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalLayerClusters"));
  desc.add<edm::InputTag>("filtered_mask", edm::InputTag("filteredLayerClusters", "iterationLabelGoesHere"));
  desc.add<edm::InputTag>("original_mask", edm::InputTag("hgcalLayerClusters", "InitialLayerClustersMask"));
  desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hgcalLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("layer_clusters_tiles", edm::InputTag("ticlLayerTileProducer"));
  desc.add<edm::InputTag>("layer_clusters_hfnose_tiles", edm::InputTag("ticlLayerTileHFNose"));
  desc.add<edm::InputTag>("seeding_regions", edm::InputTag("ticlSeedingRegionProducer"));
  desc.add<std::vector<int>>("filter_on_categories", {0});
  desc.add<double>("pid_threshold", 0.);                    // make default such that no filtering is applied
  desc.add<double>("energy_em_over_total_threshold", -1.);  // make default such that no filtering is applied
  desc.add<double>("max_longitudinal_sigmaPCA", 9999);
  desc.add<int>("shower_start_max_layer", 9999);  // make default such that no filtering is applied
  desc.add<int>("algo_verbosity", 0);
  desc.add<double>("min_cos_theta", 0.915);
  desc.add<double>("root_doublet_max_distance_from_seed_squared", 9999);
  desc.add<double>("min_cos_pointing", -1.);
  desc.add<int>("skip_layers", 0);
  desc.add<int>("max_missing_layers_in_trackster", 9999);
  desc.add<double>("etaLimitIncreaseWindow", 2.1);
  desc.add<int>("min_layers_per_trackster", 10);
  desc.add<double>("max_delta_time", 3.);  //nsigma
  desc.add<bool>("out_in_dfs", true);
  desc.add<int>("max_out_in_hops", 10);
  desc.add<bool>("oneTracksterPerTrackSeed", false);
  desc.add<bool>("promoteEmptyRegionToTrackster", false);
  desc.add<std::string>("eid_graph_path", "RecoHGCal/TICL/data/tf_models/energy_id_v0.pb");
  desc.add<std::string>("eid_input_name", "input");
  desc.add<std::string>("eid_output_name_energy", "output/regressed_energy");
  desc.add<std::string>("eid_output_name_id", "output/id_probabilities");
  desc.add<std::string>("itername", "unknown");
  desc.add<double>("eid_min_cluster_energy", 1.);
  desc.add<int>("eid_n_layers", 50);
  desc.add<int>("eid_n_clusters", 10);
  descriptions.add("trackstersProducer", desc);
}

void TrackstersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<Trackster>>();
  auto output_mask = std::make_unique<std::vector<float>>();

  const std::vector<float>& original_layerclusters_mask = evt.get(original_layerclusters_mask_token_);
  const auto& layerClusters = evt.get(clusters_token_);
  const auto& inputClusterMask = evt.get(filtered_layerclusters_mask_token_);
  const auto& layerClustersTimes = evt.get(clustersTime_token_);
  const auto& seeding_regions = evt.get(seeding_regions_token_);

  std::unordered_map<int, std::vector<int>> seedToTrackstersAssociation;
  // if it's regional iteration and there are seeding regions
  if (!seeding_regions.empty() and seeding_regions[0].index != -1) {
    auto numberOfSeedingRegions = seeding_regions.size();
    for (unsigned int i = 0; i < numberOfSeedingRegions; ++i) {
      seedToTrackstersAssociation.emplace(seeding_regions[i].index, 0);
    }
  }

  if (doNose_) {
    const auto& layer_clusters_hfnose_tiles = evt.get(layer_clusters_tiles_hfnose_token_);
    const typename PatternRecognitionAlgoBaseT<TICLLayerTilesHFNose>::Inputs inputHFNose(
        evt, es, layerClusters, inputClusterMask, layerClustersTimes, layer_clusters_hfnose_tiles, seeding_regions);

    myAlgoHFNose_->makeTracksters(inputHFNose, *result, seedToTrackstersAssociation);

  } else {
    const auto& layer_clusters_tiles = evt.get(layer_clusters_tiles_token_);
    const typename PatternRecognitionAlgoBaseT<TICLLayerTiles>::Inputs input(
        evt, es, layerClusters, inputClusterMask, layerClustersTimes, layer_clusters_tiles, seeding_regions);

    myAlgo_->makeTracksters(input, *result, seedToTrackstersAssociation);
  }
  // Now update the global mask and put it into the event
  output_mask->reserve(original_layerclusters_mask.size());
  // Copy over the previous state
  std::copy(
      std::begin(original_layerclusters_mask), std::end(original_layerclusters_mask), std::back_inserter(*output_mask));

  // Mask the used elements, accordingly
  for (auto const& trackster : *result) {
    for (auto const v : trackster.vertices()) {
      // TODO(rovere): for the moment we mask the layer cluster completely. In
      // the future, properly compute the fraction of usage.
      (*output_mask)[v] = 0.;
    }
  }

  evt.put(std::move(result));
  evt.put(std::move(output_mask));
}
