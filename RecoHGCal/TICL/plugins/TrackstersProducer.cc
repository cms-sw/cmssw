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
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"

#include "RecoHGCal/TICL/plugins/PatternRecognitionPluginFactory.h"
#include "PatternRecognitionbyCA.h"
#include "PatternRecognitionbyMultiClusters.h"

#include "PhysicsTools/TensorFlow/interface/TfGraphRecord.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "PhysicsTools/TensorFlow/interface/TfGraphDefWrapper.h"

using namespace ticl;

class TrackstersProducer : public edm::stream::EDProducer<> {
public:
  explicit TrackstersProducer(const edm::ParameterSet&);
  ~TrackstersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

  // static methods for handling the global cache
  static std::unique_ptr<TrackstersCache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(TrackstersCache*);

private:
  std::string detector_;
  bool doNose_;
  const std::string tfDnnLabel_;
  const edm::ESGetToken<TfGraphDefWrapper, TfGraphRecord> tfDnnToken_;
  const tensorflow::Session* tfSession_;
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
  ticl::Trackster::IterationIndex iterIndex_ = ticl::Trackster::IterationIndex(0);
};
DEFINE_FWK_MODULE(TrackstersProducer);

TrackstersProducer::TrackstersProducer(const edm::ParameterSet& ps)
    : detector_(ps.getParameter<std::string>("detector")),
      doNose_(detector_ == "HFNose"),
      tfDnnLabel_(ps.getParameter<std::string>("tfDnnLabel")),
      tfDnnToken_(esConsumes(edm::ESInputTag("", tfDnnLabel_))),
      tfSession_(nullptr),
      clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      filtered_layerclusters_mask_token_(consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("filtered_mask"))),
      original_layerclusters_mask_token_(consumes<std::vector<float>>(ps.getParameter<edm::InputTag>("original_mask"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("time_layerclusters"))),
      seeding_regions_token_(
          consumes<std::vector<TICLSeedingRegion>>(ps.getParameter<edm::InputTag>("seeding_regions"))),
      itername_(ps.getParameter<std::string>("itername")) {
  auto plugin = ps.getParameter<std::string>("patternRecognitionBy");
  auto pluginPSet = ps.getParameter<edm::ParameterSet>("pluginPatternRecognitionBy" + plugin);
  if (doNose_) {
    myAlgoHFNose_ = PatternRecognitionHFNoseFactory::get()->create(
        ps.getParameter<std::string>("patternRecognitionBy"), pluginPSet, consumesCollector());
    layer_clusters_tiles_hfnose_token_ =
        consumes<TICLLayerTilesHFNose>(ps.getParameter<edm::InputTag>("layer_clusters_hfnose_tiles"));
  } else {
    myAlgo_ = PatternRecognitionFactory::get()->create(
        ps.getParameter<std::string>("patternRecognitionBy"), pluginPSet, consumesCollector());
    layer_clusters_tiles_token_ = consumes<TICLLayerTiles>(ps.getParameter<edm::InputTag>("layer_clusters_tiles"));
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
  produces<std::vector<float>>();  // Mask to be applied at the next iteration
}

void TrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCAL");
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("filtered_mask", edm::InputTag("filteredLayerClusters", "iterationLabelGoesHere"));
  desc.add<edm::InputTag>("original_mask", edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask"));
  desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("layer_clusters_tiles", edm::InputTag("ticlLayerTileProducer"));
  desc.add<edm::InputTag>("layer_clusters_hfnose_tiles", edm::InputTag("ticlLayerTileHFNose"));
  desc.add<edm::InputTag>("seeding_regions", edm::InputTag("ticlSeedingRegionProducer"));
  desc.add<std::string>("patternRecognitionBy", "CA");
  desc.add<std::string>("itername", "unknown");
  desc.add<std::string>("tfDnnLabel", "tracksterSelectionTf");

  // CA Plugin
  edm::ParameterSetDescription pluginDesc;
  pluginDesc.addNode(edm::PluginDescription<PatternRecognitionFactory>("type", "CA", true));
  desc.add<edm::ParameterSetDescription>("pluginPatternRecognitionByCA", pluginDesc);
  //
  // CLUE3D Plugin
  edm::ParameterSetDescription pluginDescClue3D;
  pluginDescClue3D.addNode(edm::PluginDescription<PatternRecognitionFactory>("type", "CLUE3D", true));
  desc.add<edm::ParameterSetDescription>("pluginPatternRecognitionByCLUE3D", pluginDescClue3D);

  // FastJet Plugin
  edm::ParameterSetDescription pluginDescFastJet;
  pluginDescFastJet.addNode(edm::PluginDescription<PatternRecognitionFactory>("type", "FastJet", true));
  desc.add<edm::ParameterSetDescription>("pluginPatternRecognitionByFastJet", pluginDescFastJet);

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

  tfSession_ = es.getData(tfDnnToken_).getSession();

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
    const typename PatternRecognitionAlgoBaseT<TICLLayerTilesHFNose>::Inputs inputHFNose(evt,
                                                                                         es,
                                                                                         layerClusters,
                                                                                         inputClusterMask,
                                                                                         layerClustersTimes,
                                                                                         layer_clusters_hfnose_tiles,
                                                                                         seeding_regions,
                                                                                         tfSession_);

    myAlgoHFNose_->makeTracksters(inputHFNose, *result, seedToTrackstersAssociation);

  } else {
    const auto& layer_clusters_tiles = evt.get(layer_clusters_tiles_token_);
    const typename PatternRecognitionAlgoBaseT<TICLLayerTiles>::Inputs input(
        evt, es, layerClusters, inputClusterMask, layerClustersTimes, layer_clusters_tiles, seeding_regions, tfSession_);

    myAlgo_->makeTracksters(input, *result, seedToTrackstersAssociation);
  }
  // Now update the global mask and put it into the event
  output_mask->reserve(original_layerclusters_mask.size());
  // Copy over the previous state
  std::copy(
      std::begin(original_layerclusters_mask), std::end(original_layerclusters_mask), std::back_inserter(*output_mask));

  for (auto& trackster : *result) {
    trackster.setIteration(iterIndex_);
    // Mask the used elements, accordingly
    for (auto const v : trackster.vertices()) {
      // TODO(rovere): for the moment we mask the layer cluster completely. In
      // the future, properly compute the fraction of usage.
      (*output_mask)[v] = 0.;
    }
  }

  evt.put(std::move(result));
  evt.put(std::move(output_mask));
}
