
#include "DataFormats/TICL/interface/CaloClusterHostCollection.h"
#include "DataFormats/TICL/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/TICL/interface/ClusterMaskHost.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/alpaka/LayerClusterMergingAlgo.h"

#include <vector>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MergedLayerClustersProducer : public stream::EDProducer<> {
  public:
    MergedLayerClustersProducer(const edm::ParameterSet& config)
        : EDProducer(config), layer_cluster_mask_token_{produces()} {
      std::vector<edm::InputTag> host_tags = config.getParameter<std::vector<edm::InputTag>>("hostLayerClusters");
      for (auto& tag : host_tags) {
        host_layer_cluster_tokens_.push_back(consumes<::reco::CaloClusterHostCollection>(tag));
      }
      std::vector<edm::InputTag> device_tags = config.getParameter<std::vector<edm::InputTag>>("deviceLayerClusters");
      for (auto& tag : device_tags) {
        device_layer_cluster_tokens_.push_back(consumes(tag));
      }
    }
    ~MergedLayerClustersProducer() override = default;

    static void fillDescription(edm::ConfigurationDescriptions& description) {
      edm::ParameterSetDescription desc;

      desc.add<edm::InputTag>("deviceLayerClusters");

      description.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      auto& queue = iEvent.queue();
      std::vector<edm::Handle<::reco::CaloClusterHostCollection>> hostLayerClusters;
      // std::vector<edm::Handle<reco::CaloClusterDeviceCollection>> deviceLayerClusters;

      auto total_layer_clusters = 0;
      // auto total_associated_rechits = 0;
      for (auto token : host_layer_cluster_tokens_) {
        auto handle = iEvent.getHandle(token);
        total_layer_clusters += handle->view().position().metadata().size();
        hostLayerClusters.push_back(handle);
      }
      for (const auto& token : device_layer_cluster_tokens_) {
        const auto& prod = iEvent.get(token);
        total_layer_clusters += prod.view().position().metadata().size();
      }

      reco::CaloClusterDeviceCollection merged_layer_clusters(
          queue, total_layer_clusters, total_layer_clusters, total_layer_clusters, total_layer_clusters);

      auto start = 0u;
      for (const auto& token : device_layer_cluster_tokens_) {
        const auto& layer_clusters = iEvent.get(token);
        merging_algo.merge(iEvent.queue(), merged_layer_clusters.view(), layer_clusters.view(), start);
      }

      ticl::ClusterMaskHost layer_cluster_mask(cms::alpakatools::host(), total_layer_clusters);

      iEvent.emplace(layer_cluster_mask_token_, std::move(layer_cluster_mask));
      iEvent.emplace(merged_layer_clusters_token_, std::move(merged_layer_clusters));
    }

  private:
    std::vector<device::EDGetToken<reco::CaloClusterDeviceCollection>> device_layer_cluster_tokens_;
    std::vector<edm::EDGetTokenT<::reco::CaloClusterHostCollection>> host_layer_cluster_tokens_;
    edm::EDPutTokenT<::ticl::ClusterMaskHost> layer_cluster_mask_token_;
    device::EDPutToken<reco::CaloClusterDeviceCollection> merged_layer_clusters_token_;
    LayerClusterMerger merging_algo;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
