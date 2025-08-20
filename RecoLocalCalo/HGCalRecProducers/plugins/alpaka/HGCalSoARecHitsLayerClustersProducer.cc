#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"

#include "HGCalLayerClustersAlgoWrapper.h"

// Processes the input RecHit SoA collection and generates an output SoA
// containing all the necessary information to build the clusters.
// Specifically, this producer does not create the clusters in any format.
// Instead, it fills a SoA (HGCalSoARecHitsExtra) with the same size as the input
// RecHit SoA. This output SoA includes all the data needed to assemble the
// clusters and assigns a clusterId to each cell that belongs to a cluster.
// Consequently, this producer must be used by another downstream producer to
// either build traditional clusters or to create a SoA representing the
// clusters, complete with all required information (e.g., energy, position).
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalSoARecHitsLayerClustersProducer : public stream::EDProducer<> {
  public:
    HGCalSoARecHitsLayerClustersProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          getTokenDevice_{consumes(config.getParameter<edm::InputTag>("hgcalRecHitsSoA"))},
          deviceToken_{produces()},
          deltac_((float)config.getParameter<double>("deltac")),
          kappa_((float)config.getParameter<double>("kappa")),
          outlierDeltaFactor_((float)config.getParameter<double>("outlierDeltaFactor")) {}

    ~HGCalSoARecHitsLayerClustersProducer() override = default;

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto const& deviceInput = iEvent.get(getTokenDevice_);
      //std::cout << "Size of device collection: " << deviceInput->metadata().size() << std::endl;
      auto const input_v = deviceInput.view();
      // Allocate output SoA
      HGCalSoARecHitsExtraDeviceCollection output(deviceInput->metadata().size(), iEvent.queue());
      auto output_v = output.view();

      algo_.run(
          iEvent.queue(), deviceInput->metadata().size(), deltac_, kappa_, outlierDeltaFactor_, input_v, output_v);
      iEvent.emplace(deviceToken_, std::move(output));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("hgcalRecHitsSoA", edm::InputTag("TO BE DEFINED"));
      desc.add<double>("deltac", 1.3);
      desc.add<double>("kappa", 9.);
      desc.add<double>("outlierDeltaFactor", 2.);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    // use device::EDGetToken<T> to read from device memory space
    device::EDGetToken<HGCalSoARecHitsDeviceCollection> const getTokenDevice_;
    device::EDPutToken<HGCalSoARecHitsExtraDeviceCollection> const deviceToken_;
    HGCalLayerClustersAlgoWrapper algo_;
    const float deltac_;
    const float kappa_;
    const float outlierDeltaFactor_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSoARecHitsLayerClustersProducer);
