#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/HGCalReco/interface/HGCalSoAClusters.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "RecoHGCal/TICL/interface/alpaka/PatternRecognitionAlgoBase.h"
#include "RecoHGCal/TICL/plugins/alpaka/PatternRecognitionByCLUEstering.h"
#include "CLUEstering/CLUEstering.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HeterogeneousTracksterProducer : public stream::EDProducer<> {
  public:
    HeterogeneousTracksterProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          deviceTokenSoAClusters_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          legacyTrackstersToken_{produces()},
          algo_{std::make_unique<PatternRecognitionByCLUEstering>(config)} {}
    ~HeterogeneousTracksterProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalRecHitsLayerClustersSoA"));
	  // Still needed? I guess I still need to save them here and then copy to algo
      desc.add<double>("rho_c", 0.6);
      desc.add<std::vector<double>>("dc", {2., 2., 2});
      desc.add<std::vector<double>>("dm", {1.8, 1.8, 2});
      desc.add<bool>("verbose", false);
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      const auto& lc = iEvent.get(deviceTokenSoAClusters_);
      auto tracksters = std::vector<ticl::Trackster>();
      auto& queue = iEvent.queue();
      algo_->makeTracksters(queue, lc, tracksters);

      iEvent.emplace(legacyTrackstersToken_, std::move(tracksters));
    }

  private:
    device::EDGetToken<HGCalSoAClustersDeviceCollection> const deviceTokenSoAClusters_;
    edm::EDPutTokenT<std::vector<ticl::Trackster>> const legacyTrackstersToken_;
    std::unique_ptr<PatternRecognitionAlgoBase> algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HeterogeneousTracksterProducer);
