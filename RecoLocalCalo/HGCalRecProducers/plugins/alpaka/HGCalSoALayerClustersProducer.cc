#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoAClusters.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalSoAClustersExtra.h"
#include "HGCalLayerClustersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalSoALayerClustersProducer : public stream::SynchronizingEDProducer<> {
  public:
    HGCalSoALayerClustersProducer(edm::ParameterSet const& config)
        : getTokenDeviceRecHits_{consumes(config.getParameter<edm::InputTag>("hgcalRecHitsSoA"))},
          getTokenDeviceClusters_{consumes(config.getParameter<edm::InputTag>("hgcalRecHitsLayerClustersSoA"))},
          deviceTokenSoAClusters_{produces()},
          thresholdW0_(config.getParameter<double>("thresholdW0")),
          positionDeltaRho2_(config.getParameter<double>("positionDeltaRho2")) {}

    ~HGCalSoALayerClustersProducer() override = default;

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      // Get LayerClusters almost-SoA on device: this has still the same
      // cardinality as the RecHitsSoA, but has all the required information
      // to assemble the clusters, i.e., it has the cluster index assigned to
      // each rechit.
      auto const& deviceInputClusters = iEvent.get(getTokenDeviceClusters_);
      auto const inputClusters_v = deviceInputClusters.view();
      //
      // Allocate output SoA for the clusters, one entry for each cluster
      auto device_numclusters = cms::alpakatools::make_device_view<const unsigned int>(
          alpaka::getDev(iEvent.queue()), inputClusters_v.numberOfClustersScalar());
      auto host_numclusters = cms::alpakatools::make_host_view<unsigned int>(num_clusters_);
      alpaka::memcpy(iEvent.queue(), host_numclusters, device_numclusters);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      // Get RecHitsSoA on the device
      auto const& deviceInputRecHits = iEvent.get(getTokenDeviceRecHits_);
      auto const inputRechits_v = deviceInputRecHits.view();

      // Get LayerClusters almost-SoA on device: this has still the same
      // cardinality as the RecHitsSoA, but has all the required information
      // to assemble the clusters, i.e., it has the cluster index assigned to
      // each rechit.
      auto const& deviceInputClusters = iEvent.get(getTokenDeviceClusters_);
      auto const inputClusters_v = deviceInputClusters.view();

      ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<HGCalSoAClusters> output(num_clusters_, iEvent.queue());
      auto output_v = output.view();
      // Allocate service SoA cluster
      ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<HGCalSoAClustersExtra> outputService(num_clusters_,
                                                                                            iEvent.queue());
      auto output_service_v = outputService.view();

      algo_.run(iEvent.queue(),
                num_clusters_,
                thresholdW0_,
                positionDeltaRho2_,
                inputRechits_v,
                inputClusters_v,
                output_v,
                output_service_v);
      iEvent.emplace(deviceTokenSoAClusters_, std::move(output));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("hgcalRecHitsLayerClustersSoA", edm::InputTag("TO BE DEFINED"));
      desc.add<edm::InputTag>("hgcalRecHitsSoA", edm::InputTag("TO BE DEFINED"));
      desc.add<double>("thresholdW0", 2.9);
      desc.add<double>("positionDeltaRho2", 1.69);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    device::EDGetToken<ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<HGCalSoARecHits>> const getTokenDeviceRecHits_;
    device::EDGetToken<ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<HGCalSoARecHitsExtra>> const
        getTokenDeviceClusters_;
    device::EDPutToken<ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<HGCalSoAClusters>> const deviceTokenSoAClusters_;
    HGCalLayerClustersSoAAlgoWrapper algo_;
    unsigned int num_clusters_;
    float thresholdW0_;
    float positionDeltaRho2_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSoALayerClustersProducer);
