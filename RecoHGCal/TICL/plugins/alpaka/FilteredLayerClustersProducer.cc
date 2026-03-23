#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/TICL/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/ClusterMaskDevice.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include "RecoHGCal/TICL/plugins/alpaka/ClusterFilterByAlgoAndSize.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  class HeterogeneousFilteredLayerClustersProducer : public stream::EDProducer<> {
  public:
    HeterogeneousFilteredLayerClustersProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          deviceTokenSoAClusters_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          clustersMaskToken_{consumes(config.getParameter<edm::InputTag>("layerClustersInputMask"))},
          filteredClustersMaskToken_{produces()},
          minClusterSize_{config.getParameter<unsigned int>("minClusterSize")},
          maxClusterSize_{config.getParameter<unsigned int>("maxClusterSize")},
          algo_(config) {}
    ~HeterogeneousFilteredLayerClustersProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalSoALayerClusters"));
      desc.add<edm::InputTag>("layerClustersInputMask",
                              edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask"));
      desc.add<unsigned int>("minClusterSize");
      desc.add<unsigned int>("maxClusterSize");

      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      const auto& layerClusters = iEvent.get(deviceTokenSoAClusters_);
      const auto& layerClustersMask = iEvent.get(clustersMaskToken_);

      auto& queue = iEvent.queue();
      ticl::ClusterMaskDevice filteredMask(queue, layerClustersMask.view().metadata().size());
      alpaka::memcpy(queue, filteredMask.buffer(), layerClustersMask.buffer());
      algo_.filter(queue, layerClusters, filteredMask, minClusterSize_, maxClusterSize_);

      iEvent.emplace(filteredClustersMaskToken_, std::move(filteredMask));
    }

  private:
    device::EDGetToken<reco::CaloClusterDeviceCollection> const deviceTokenSoAClusters_;
    device::EDGetToken<ticl::ClusterMaskDevice> clustersMaskToken_;
    device::EDPutToken<ticl::ClusterMaskDevice> filteredClustersMaskToken_;

    const uint32_t minClusterSize_;
    const uint32_t maxClusterSize_;

    ClusterFilterByAlgoAndSize algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl
