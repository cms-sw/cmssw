#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Copies a rechit SoA with its rechits grouped by layer, in increasing layer order and in their order within a
  // layer, and produces the number of rechits of each layer that has some ("layerSizes"), as the CLUEstering layer
  // clustering needs them. For rechit SoAs that are not grouped by layer, as the ones of the test beam.
  class HGCalSoARecHitsByLayerProducer : public stream::SynchronizingEDProducer<> {
  public:
    HGCalSoARecHitsByLayerProducer(const edm::ParameterSet& config)
        : SynchronizingEDProducer(config),
          recHitsToken_{consumes(config.getParameter<edm::InputTag>("recHits"))},
          recHitsByLayerToken_{produces()},
          layerSizesToken_{produces("layerSizes")} {}

    // The rechits are sorted on the host
    void acquire(const device::Event& iEvent, const device::EventSetup&) override {
      const auto& recHits = iEvent.get(recHitsToken_);
      hostRecHits_.emplace(iEvent.queue(), recHits->metadata().size());
      alpaka::memcpy(iEvent.queue(), hostRecHits_->buffer(), recHits.const_buffer());
    }

    void produce(device::Event& iEvent, const device::EventSetup&) override {
      const auto input = hostRecHits_->const_view();
      const int32_t nRecHits = input.metadata().size();
      std::vector<int32_t> order(nRecHits);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(
          order.begin(), order.end(), [&input](int32_t a, int32_t b) { return input[a].layer() < input[b].layer(); });

      HGCalSoARecHitsHostCollection recHitsByLayer(iEvent.queue(), nRecHits);
      auto output = recHitsByLayer.view();
      std::vector<uint32_t> layerSizes;
      for (int32_t i = 0; i < nRecHits; ++i) {
        output[i] = input[order[i]];
        if (i == 0 or input[order[i]].layer() != input[order[i - 1]].layer()) {
          layerSizes.push_back(0);
        }
        ++layerSizes.back();
      }
      hostRecHits_.reset();

      if constexpr (std::is_same_v<Device, alpaka_common::DevHost>) {
        iEvent.emplace(recHitsByLayerToken_, std::move(recHitsByLayer));
      } else {
        HGCalSoARecHitsDeviceCollection deviceRecHits(iEvent.queue(), nRecHits);
        alpaka::memcpy(iEvent.queue(), deviceRecHits.buffer(), recHitsByLayer.const_buffer());
        iEvent.emplace(recHitsByLayerToken_, std::move(deviceRecHits));
      }
      iEvent.emplace(layerSizesToken_, std::move(layerSizes));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("recHits", edm::InputTag("hgcalSoARecHits"));
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<HGCalSoARecHitsDeviceCollection> recHitsToken_;
    const device::EDPutToken<HGCalSoARecHitsDeviceCollection> recHitsByLayerToken_;
    const edm::EDPutTokenT<std::vector<uint32_t>> layerSizesToken_;
    // The input rechits, copied to the host in acquire
    std::optional<HGCalSoARecHitsHostCollection> hostRecHits_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSoARecHitsByLayerProducer);
