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

  class HGCalCLUEsteringLayerClustersProducer : public stream::EDProducer<> {
  public:
    HGCalCLUEsteringLayerClustersProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          getTokenDevice_{consumes(config.getParameter<edm::InputTag>("hgcalRecHitsSoA"))},
          getTokenLayerSizes_{consumes<std::vector<uint32_t>>(
              edm::InputTag(config.getParameter<edm::InputTag>("hgcalRecHitsSoA").label(), "layerSizes"))},
          deviceToken_{produces()},
          deltac_(config.getParameter<float>("deltac")),
          kappa_(config.getParameter<float>("kappa")),
          outlierDeltaFactor_(config.getParameter<float>("outlierDeltaFactor")),
          isScintillator_(config.getParameter<std::string>("detector") == "BH") {}

    ~HGCalCLUEsteringLayerClustersProducer() override = default;

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto const& deviceInput = iEvent.get(getTokenDevice_);
      auto const input_v = deviceInput.view();
      // Per-layer batch sizes computed upstream by the rechit producer (the SoA
      // is emitted layer-contiguous), used directly as the CLUEstering batch sizes.
      auto const& layerSizes = iEvent.get(getTokenLayerSizes_);
      // Allocate output SoA, same size as the input RecHit SoA.
      HGCalSoARecHitsExtraDeviceCollection output(iEvent.queue(), deviceInput->metadata().size());
      auto output_v = output.view();

      algo_.run(iEvent.queue(),
                deviceInput->metadata().size(),
                deltac_,
                kappa_,
                outlierDeltaFactor_,
                isScintillator_,
                std::span<const uint32_t>(layerSizes),
                input_v,
                output_v);
      iEvent.emplace(deviceToken_, std::move(output));
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
    device::EDGetToken<HGCalSoARecHitsDeviceCollection> const getTokenDevice_;
    edm::EDGetTokenT<std::vector<uint32_t>> const getTokenLayerSizes_;
    device::EDPutToken<HGCalSoARecHitsExtraDeviceCollection> const deviceToken_;
    HGCalCLUEsteringAlgoWrapper algo_;
    const float deltac_;
    const float kappa_;
    const float outlierDeltaFactor_;
    const bool isScintillator_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalCLUEsteringLayerClustersProducer);
