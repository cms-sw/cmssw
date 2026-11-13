#include <string>
#include <utility>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "HGCalLayerClustersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Builds the layer clusters of one HGCAL component (energy, number of cells, seed, position and time) from its
  // rechits and their clustering (the layer cluster and the seed flag of each rechit)
  class HGCalSoALayerClustersProducer : public stream::SynchronizingEDProducer<> {
  public:
    HGCalSoALayerClustersProducer(edm::ParameterSet const& config)
        : SynchronizingEDProducer(config),
          recHitsToken_{consumes(config.getParameter<edm::InputTag>("hgcalRecHitsSoA"))},
          recHitsClusteringToken_{consumes(config.getParameter<edm::InputTag>("hgcalRecHitsLayerClustersSoA"))},
          layerClustersToken_{produces()},
          thresholdW0_(config.getParameter<float>("thresholdW0")),
          positionDeltaRho2_(config.getParameter<float>("positionDeltaRho2")),
          isScintillator_(config.getParameter<std::string>("detector") == "BH"),
          nHitsTime_(config.getParameter<unsigned int>("nHitsTime")) {}

    // The number of layer clusters is on the device, and the host needs it to allocate them
    void acquire(const device::Event& iEvent, const device::EventSetup&) override {
      const auto recHitsClustering = iEvent.get(recHitsClusteringToken_).view();
      alpaka::memcpy(iEvent.queue(),
                     cms::alpakatools::make_host_view<unsigned int>(nLayerClusters_),
                     cms::alpakatools::make_device_view<const unsigned int>(
                         alpaka::getDev(iEvent.queue()), recHitsClustering.numberOfClustersScalar()));
    }

    void produce(device::Event& iEvent, const device::EventSetup&) override {
      reco::CaloClusterDeviceCollection layerClusters(
          iEvent.queue(), nLayerClusters_, nLayerClusters_, nLayerClusters_, nLayerClusters_);
      algo_.run(iEvent.queue(),
                nLayerClusters_,
                thresholdW0_,
                positionDeltaRho2_,
                isScintillator_,
                nHitsTime_,
                iEvent.get(recHitsToken_).view(),
                iEvent.get(recHitsClusteringToken_).view(),
                layerClusters.view());
      iEvent.emplace(layerClustersToken_, std::move(layerClusters));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("hgcalRecHitsLayerClustersSoA", edm::InputTag("TO BE DEFINED"));
      desc.add<edm::InputTag>("hgcalRecHitsSoA", edm::InputTag("TO BE DEFINED"));
      desc.add<float>("thresholdW0", 2.9);
      desc.add<float>("positionDeltaRho2", 1.69);
      desc.add<std::string>("detector", "EE")
          ->setComment("HGCAL component; 'BH' computes the scintillator centroid in cartesian (eta,phi->x,y).");
      desc.add<unsigned int>("nHitsTime", 3)
          ->setComment("Minimum number of rechits with a time for a layer cluster to get a time.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<HGCalSoARecHitsDeviceCollection> recHitsToken_;
    const device::EDGetToken<HGCalSoARecHitsExtraDeviceCollection> recHitsClusteringToken_;
    const device::EDPutToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    const float thresholdW0_;
    const float positionDeltaRho2_;
    const bool isScintillator_;
    const unsigned int nHitsTime_;
    const HGCalLayerClustersSoAAlgoWrapper algo_{};
    // copied from the device in acquire, used in produce
    unsigned int nLayerClusters_ = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSoALayerClustersProducer);
