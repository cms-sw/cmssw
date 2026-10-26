#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "HGCalCLUEsteringAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Clusters the rechits of one HGCAL component, layer by layer, with CLUEstering: produces the layer cluster and the
  // seed flag of each rechit, and the number of layer clusters
  class HGCalCLUEsteringLayerClustersProducer : public stream::EDProducer<> {
  public:
    HGCalCLUEsteringLayerClustersProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          recHitsToken_{consumes(config.getParameter<edm::InputTag>("hgcalRecHitsSoA"))},
          layerSizesToken_{consumes<std::vector<uint32_t>>(
              edm::InputTag(config.getParameter<edm::InputTag>("hgcalRecHitsSoA").label(), "layerSizes"))},
          recHitsClusteringToken_{produces()},
          deltac_(config.getParameter<float>("deltac")),
          kappa_(config.getParameter<float>("kappa")),
          outlierDeltaFactor_(config.getParameter<float>("outlierDeltaFactor")),
          isScintillator_(config.getParameter<std::string>("detector") == "BH") {}

    void produce(device::Event& iEvent, const device::EventSetup&) override {
      const auto& recHits = iEvent.get(recHitsToken_);
      // Per-layer batch sizes computed upstream by the rechit producer (the SoA
      // is emitted layer-contiguous), used directly as the CLUEstering batch sizes.
      const auto& layerSizes = iEvent.get(layerSizesToken_);
      // One row per rechit, like the rechit SoA
      HGCalSoARecHitsExtraDeviceCollection recHitsClustering(iEvent.queue(), recHits->metadata().size());

      algo_.run(iEvent.queue(),
                recHits->metadata().size(),
                deltac_,
                kappa_,
                outlierDeltaFactor_,
                isScintillator_,
                std::span<const uint32_t>(layerSizes),
                recHits.view(),
                recHitsClustering.view());
      iEvent.emplace(recHitsClusteringToken_, std::move(recHitsClustering));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("hgcalRecHitsSoA", edm::InputTag("TO BE DEFINED"));
      desc.add<std::string>("detector", "EE")
          ->setComment("HGCAL component; 'BH' selects the periodic (eta,phi) scintillator metric.");
      desc.add<float>("deltac", 1.3);
      desc.add<float>("kappa", 9.);
      desc.add<float>("outlierDeltaFactor", 2.);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<HGCalSoARecHitsDeviceCollection> recHitsToken_;
    const edm::EDGetTokenT<std::vector<uint32_t>> layerSizesToken_;
    const device::EDPutToken<HGCalSoARecHitsExtraDeviceCollection> recHitsClusteringToken_;
    const float deltac_;
    const float kappa_;
    const float outlierDeltaFactor_;
    const bool isScintillator_;
    const HGCalCLUEsteringAlgoWrapper algo_{};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalCLUEsteringLayerClustersProducer);
