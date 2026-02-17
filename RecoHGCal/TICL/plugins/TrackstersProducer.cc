// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch,marco.rovere@cern.ch
// Date: 09/2018

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"
#include "RecoHGCal/TICL/plugins/PatternRecognitionPluginFactory.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

using namespace ticl;

class TrackstersProducer : public edm::stream::EDProducer<edm::GlobalCache<ticl::TICLONNXGlobalCache>> {
public:
  explicit TrackstersProducer(const edm::ParameterSet&, ticl::TICLONNXGlobalCache const* cache);
  ~TrackstersProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::unique_ptr<ticl::TICLONNXGlobalCache> initializeGlobalCache(const edm::ParameterSet& iConfig);
  static void globalEndJob(ticl::TICLONNXGlobalCache const*);

  void produce(edm::Event&, const edm::EventSetup&) override;

  void beginRun(const edm::Run&, const edm::EventSetup& es) override {
    const auto& geom = es.getData(geometry_token_);
    rhtools_.setGeometry(geom);

    // Configure the pattern recognition plugin once per run/IOV.
    if (doNose_) {
      myAlgoHFNose_->setGeometry(rhtools_);
    } else {
      myAlgo_->setGeometry(rhtools_);
    }
  }

private:
  std::string detector_;
  bool doNose_;
  std::unique_ptr<PatternRecognitionAlgoBaseT<TICLLayerTiles>> myAlgo_;
  std::unique_ptr<PatternRecognitionAlgoBaseT<TICLLayerTilesHFNose>> myAlgoHFNose_;

  std::unique_ptr<TracksterInferenceAlgoBase> inferenceAlgo_;

  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  const edm::EDGetTokenT<std::vector<float>> filtered_layerclusters_mask_token_;
  const edm::EDGetTokenT<std::vector<float>> original_layerclusters_mask_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;

  edm::EDGetTokenT<TICLLayerTiles> layer_clusters_tiles_token_;
  edm::EDGetTokenT<TICLLayerTilesHFNose> layer_clusters_tiles_hfnose_token_;
  const edm::EDGetTokenT<std::vector<TICLSeedingRegion>> seeding_regions_token_;

  ticl::Trackster::IterationIndex iterIndex_ = ticl::Trackster::IterationIndex(0);

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  hgcal::RecHitTools rhtools_;
  const std::string itername_;
};

DEFINE_FWK_MODULE(TrackstersProducer);

TrackstersProducer::TrackstersProducer(const edm::ParameterSet& ps, ticl::TICLONNXGlobalCache const* cache)
    : detector_(ps.getParameter<std::string>("detector")),
      doNose_(detector_ == "HFNose"),
      clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      filtered_layerclusters_mask_token_(consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("filtered_mask"))),
      original_layerclusters_mask_token_(consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("original_mask"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("time_layerclusters"))),
      seeding_regions_token_(
          consumes<std::vector<TICLSeedingRegion>>(ps.getParameter<edm::InputTag>("seeding_regions"))),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      itername_(ps.getParameter<std::string>("itername")) {
  const auto plugin = ps.getParameter<std::string>("patternRecognitionBy");
  const auto pluginPSet = ps.getParameter<edm::ParameterSet>("pluginPatternRecognitionBy" + plugin);

  if (doNose_) {
    myAlgoHFNose_ = PatternRecognitionHFNoseFactory::get()->create(plugin, pluginPSet, consumesCollector());
    layer_clusters_tiles_hfnose_token_ =
        consumes<TICLLayerTilesHFNose>(ps.getParameter<edm::InputTag>("layer_clusters_hfnose_tiles"));
  } else {
    myAlgo_ = PatternRecognitionFactory::get()->create(plugin, pluginPSet, consumesCollector());
    layer_clusters_tiles_token_ = consumes<TICLLayerTiles>(ps.getParameter<edm::InputTag>("layer_clusters_tiles"));
  }

  // Instantiate the inference plugin only if it is configured with at least one non-empty model path.
  const std::string inferencePlugin = ps.getParameter<std::string>("inferenceAlgo");
  if (!inferencePlugin.empty()) {
    const edm::ParameterSet inferencePSet = ps.getParameter<edm::ParameterSet>("pluginInferenceAlgo" + inferencePlugin);

    const bool hasSingleModel = inferencePSet.existsAs<std::string>("onnxModelPath", true) &&
                                !inferencePSet.getParameter<std::string>("onnxModelPath").empty();
    const bool hasPIDModel = inferencePSet.existsAs<std::string>("onnxPIDModelPath", true) &&
                             !inferencePSet.getParameter<std::string>("onnxPIDModelPath").empty();
    const bool hasEnergyModel = inferencePSet.existsAs<std::string>("onnxEnergyModelPath", true) &&
                                !inferencePSet.getParameter<std::string>("onnxEnergyModelPath").empty();

    if (hasSingleModel || hasPIDModel || hasEnergyModel) {
      inferenceAlgo_ = TracksterInferenceAlgoFactory::get()->create(inferencePlugin, inferencePSet, cache);
    }
  }

  if (itername_ == "TrkEM")
    iterIndex_ = ticl::Trackster::TRKEM;
  else if (itername_ == "EM")
    iterIndex_ = ticl::Trackster::EM;
  else if (itername_ == "Trk")
    iterIndex_ = ticl::Trackster::TRKHAD;
  else if (itername_ == "HAD")
    iterIndex_ = ticl::Trackster::HAD;
  else if (itername_ == "MIP")
    iterIndex_ = ticl::Trackster::MIP;

  produces<std::vector<Trackster>>();
  produces<std::vector<float>>();
}

std::unique_ptr<ticl::TICLONNXGlobalCache> TrackstersProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return ticl::TICLONNXGlobalCache::initialize(iConfig);
}

void TrackstersProducer::globalEndJob(ticl::TICLONNXGlobalCache const*) {}

void TrackstersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<Trackster>>();
  auto initialResult = std::make_unique<std::vector<Trackster>>();
  auto output_mask = std::make_unique<std::vector<float>>();

  const auto& original_layerclusters_mask = evt.get(original_layerclusters_mask_token_);
  const auto& layerClusters = evt.get(clusters_token_);
  const auto& inputClusterMask = evt.get(filtered_layerclusters_mask_token_);
  const auto& layerClustersTimes = evt.get(clustersTime_token_);
  const auto& seeding_regions = evt.get(seeding_regions_token_);

  std::unordered_map<int, std::vector<int>> seedToTrackstersAssociation;

  if (!seeding_regions.empty() && seeding_regions[0].index != -1) {
    for (unsigned int i = 0; i < seeding_regions.size(); ++i) {
      seedToTrackstersAssociation.emplace(seeding_regions[i].index, 0);
    }
  }

  if (!seeding_regions.empty()) {
    if (doNose_) {
      const auto& tiles = evt.get(layer_clusters_tiles_hfnose_token_);
      const typename PatternRecognitionAlgoBaseT<TICLLayerTilesHFNose>::Inputs inputHFNose(
          evt, es, layerClusters, inputClusterMask, layerClustersTimes, tiles, seeding_regions);

      myAlgoHFNose_->makeTracksters(inputHFNose, *initialResult, seedToTrackstersAssociation);

      if (inferenceAlgo_) {
        inferenceAlgo_->runInference(layerClusters, *initialResult, rhtools_);
      }

      myAlgoHFNose_->filter(*result, *initialResult, inputHFNose, seedToTrackstersAssociation);
    } else {
      const auto& tiles = evt.get(layer_clusters_tiles_token_);
      const typename PatternRecognitionAlgoBaseT<TICLLayerTiles>::Inputs input(
          evt, es, layerClusters, inputClusterMask, layerClustersTimes, tiles, seeding_regions);

      myAlgo_->makeTracksters(input, *initialResult, seedToTrackstersAssociation);

      if (inferenceAlgo_) {
        inferenceAlgo_->runInference(layerClusters, *initialResult, rhtools_);
      }

      myAlgo_->filter(*result, *initialResult, input, seedToTrackstersAssociation);
    }
  }

  output_mask->reserve(original_layerclusters_mask.size());
  std::copy(original_layerclusters_mask.begin(), original_layerclusters_mask.end(), std::back_inserter(*output_mask));

  for (auto& trackster : *result) {
    trackster.setIteration(iterIndex_);
    for (auto const v : trackster.vertices()) {
      (*output_mask)[v] = 0.f;
    }
  }

  evt.put(std::move(result));
  evt.put(std::move(output_mask));
}

void TrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("filtered_mask", edm::InputTag("filteredLayerClusters", "iterationLabelGoesHere"));
  desc.add<edm::InputTag>("original_mask", edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask"));
  desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("layer_clusters_tiles", edm::InputTag("ticlLayerTileProducer"));
  desc.add<edm::InputTag>("layer_clusters_hfnose_tiles", edm::InputTag("ticlLayerTileHFNose"));
  desc.add<edm::InputTag>("seeding_regions", edm::InputTag("ticlSeedingRegionProducer"));
  desc.add<std::string>("patternRecognitionBy", "CLUE3D");
  desc.add<std::string>("itername", "unknown");

  // Inference plugin name (can be left as default). If the corresponding plugin PSet
  // contains only empty model path strings, inference will be disabled at runtime.
  desc.add<std::string>("inferenceAlgo", "");

  // Pattern recognition plugins
  edm::ParameterSetDescription pluginDescCA;
  pluginDescCA.addNode(edm::PluginDescription<PatternRecognitionFactory>("type", "CA", true));
  desc.add<edm::ParameterSetDescription>("pluginPatternRecognitionByCA", pluginDescCA);

  edm::ParameterSetDescription pluginDescClue3D;
  pluginDescClue3D.addNode(edm::PluginDescription<PatternRecognitionFactory>("type", "CLUE3D", true));
  desc.add<edm::ParameterSetDescription>("pluginPatternRecognitionByCLUE3D", pluginDescClue3D);

  edm::ParameterSetDescription pluginDescFastJet;
  pluginDescFastJet.addNode(edm::PluginDescription<PatternRecognitionFactory>("type", "FastJet", true));
  desc.add<edm::ParameterSetDescription>("pluginPatternRecognitionByFastJet", pluginDescFastJet);

  edm::ParameterSetDescription pluginDescRecovery;
  pluginDescRecovery.addNode(edm::PluginDescription<PatternRecognitionFactory>("type", "Recovery", true));
  desc.add<edm::ParameterSetDescription>("pluginPatternRecognitionByRecovery", pluginDescRecovery);

  // Inference plugins
  edm::ParameterSetDescription inferenceDescDNN;
  inferenceDescDNN.addNode(
      edm::PluginDescription<TracksterInferenceAlgoFactory>("type", "TracksterInferenceByDNN", true));
  desc.add<edm::ParameterSetDescription>("pluginInferenceAlgoTracksterInferenceByDNN", inferenceDescDNN);

  edm::ParameterSetDescription inferenceDescPFN;
  inferenceDescPFN.addNode(
      edm::PluginDescription<TracksterInferenceAlgoFactory>("type", "TracksterInferenceByPFN", true));
  desc.add<edm::ParameterSetDescription>("pluginInferenceAlgoTracksterInferenceByPFN", inferenceDescPFN);

  descriptions.add("trackstersProducer", desc);
}
