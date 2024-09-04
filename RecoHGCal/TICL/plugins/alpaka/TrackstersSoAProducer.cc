#include <iostream>
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

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include "DataFormats/HGCalReco/interface/alpaka/TracksterSoADeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersFilteredMaskDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBaseSoA.h"
#include "RecoHGCal/TICL/plugins/alpaka/PatternRecognitionByCLUE3DSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackstersSoAProducer : public stream::EDProducer<> {
    public:
      TrackstersSoAProducer(edm::ParameterSet const& config)
      : deviceTokenSoAClusters_(consumes(config.getParameter<edm::InputTag>("layer_clusters"))),
        cluster_mask_token_(consumes(config.getParameter<edm::InputTag>("filtered_clusters_mask"))),
        layer_clusters_tiles_token_(consumes(config.getParameter<edm::InputTag>("layer_clusters_tiles"))),
        seeding_regions_token_(consumes(config.getParameter<edm::InputTag>("seeding_regions"))),
        tracksterSoA_token_{produces()},
        clue3d(std::make_unique<PatternRecognitionByCLUE3DSoA<TICLLayerTiles>>(config, consumesCollector()))
        {}

      ~TrackstersSoAProducer() override = default;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hltHgcalSoALayerClustersProducer"));
        desc.add<edm::InputTag>("filtered_clusters_mask", edm::InputTag("hltFilteredLayerClustersSoAProducer"));
        desc.add<edm::InputTag>("layer_clusters_tiles", edm::InputTag("ticlLayerTileProducer"));
        desc.add<edm::InputTag>("seeding_regions", edm::InputTag("ticlSeedingRegionProducer"));
        descriptions.addWithDefaultLabel(desc);
      }

      void produce(device::Event& evt, device::EventSetup const& es) override {
        const auto& layer_clusters = evt.get(deviceTokenSoAClusters_);
        const auto layerClustersSoAConstView = layer_clusters.view();
        const auto& cluster_masks = evt.get(cluster_mask_token_);
        const auto clusterMasksConstView = cluster_masks.view();
        const auto& seeding_regions = evt.get(seeding_regions_token_);
        const auto& layer_cluster_tiles = evt.get(layer_clusters_tiles_token_);

        const PatternRecognitionAlgoBaseSoAT<TICLLayerTiles>::Inputs inputs(
          layerClustersSoAConstView, clusterMasksConstView, layer_cluster_tiles, seeding_regions);
        clue3d->makeTracksters(evt.queue(), inputs);
      }

    private:

      const device::EDGetToken<HGCalSoAClustersDeviceCollection> deviceTokenSoAClusters_;
      const device::EDGetToken<HGCalSoAClustersFilteredMaskDeviceCollection> cluster_mask_token_;
      edm::EDGetTokenT<TICLLayerTiles> layer_clusters_tiles_token_;
      const edm::EDGetTokenT<std::vector<TICLSeedingRegion>> seeding_regions_token_;
      device::EDPutToken<TracksterSoADeviceCollection> tracksterSoA_token_;
      std::unique_ptr<PatternRecognitionByCLUE3DSoA<TICLLayerTiles>> clue3d;
  };

}

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TrackstersSoAProducer);