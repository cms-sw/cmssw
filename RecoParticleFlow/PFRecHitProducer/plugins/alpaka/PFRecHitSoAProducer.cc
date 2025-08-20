#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsRecord.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyRecord.h"

#include "CalorimeterDefinitions.h"
#include "PFRecHitProducerKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace particleFlowRecHitProducer;

  template <typename CAL>
  class PFRecHitSoAProducer : public stream::SynchronizingEDProducer<> {
  public:
    PFRecHitSoAProducer(edm::ParameterSet const& config)
        : SynchronizingEDProducer(config),
          topologyToken_(esConsumes(config.getParameter<edm::ESInputTag>("topology"))),
          pfRecHitsToken_(produces()),
          sizeToken_(produces()),
          synchronise_(config.getUntrackedParameter<bool>("synchronise")),
          size_{cms::alpakatools::make_host_buffer<uint32_t, Platform>()} {
      const std::vector<edm::ParameterSet> producers = config.getParameter<std::vector<edm::ParameterSet>>("producers");
      recHitsToken_.reserve(producers.size());
      for (const edm::ParameterSet& producer : producers) {
        recHitsToken_.emplace_back(consumes(producer.getParameter<edm::InputTag>("src")),
                                   esConsumes(producer.getParameter<edm::ESInputTag>("params")));
      }
    }

    void acquire(device::Event const& event, const device::EventSetup& setup) override {
      const typename CAL::TopologyTypeDevice& topology = setup.getData(topologyToken_);

      uint32_t num_recHits = 0;
      for (const auto& token : recHitsToken_)
        num_recHits += event.get(token.first)->metadata().size();

      pfRecHits_.emplace((int)num_recHits, event.queue());
      *size_ = 0;

      if (num_recHits != 0) {
        PFRecHitProducerKernel<CAL> kernel{event.queue(), num_recHits};
        for (const auto& token : recHitsToken_)
          kernel.processRecHits(
              event.queue(), event.get(token.first), setup.getData(token.second), topology, *pfRecHits_);
        kernel.associateTopologyInfo(event.queue(), topology, *pfRecHits_);
        auto size_d = cms::alpakatools::make_device_view<uint32_t>(event.queue(), pfRecHits_->view().size());
        alpaka::memcpy(event.queue(), size_, size_d);
      }
    }

    void produce(device::Event& event, const device::EventSetup& setup) override {
      event.emplace(pfRecHitsToken_, std::move(*pfRecHits_));
      event.emplace(sizeToken_, *size_);
      pfRecHits_.reset();

      if (synchronise_)
        alpaka::wait(event.queue());
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      edm::ParameterSetDescription producers;
      producers.add<edm::InputTag>("src", edm::InputTag(""))->setComment("Input CaloRecHitSoA");
      producers.add<edm::ESInputTag>("params", edm::ESInputTag(""))->setComment("Quality cut parameters");
      std::vector<edm::ParameterSet> producersDefault(1);
      producersDefault[0].addParameter<edm::InputTag>("src", edm::InputTag(""));
      producersDefault[0].addParameter<edm::ESInputTag>("params", edm::ESInputTag(""));
      desc.addVPSet("producers", producers, producersDefault)->setComment("List of inputs and quality cuts");
      desc.add<edm::ESInputTag>("topology", edm::ESInputTag(""))->setComment("Topology information");
      desc.addUntracked<bool>("synchronise", false)
          ->setComment("Add synchronisation point after execution (for benchmarking asynchronous execution)");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    // module configuration
    const device::ESGetToken<typename CAL::TopologyTypeDevice, typename CAL::TopologyRecordType> topologyToken_;
    std::vector<std::pair<device::EDGetToken<typename CAL::CaloRecHitSoATypeDevice>,
                          device::ESGetToken<typename CAL::ParameterType, typename CAL::ParameterRecordType>>>
        recHitsToken_;
    const device::EDPutToken<reco::PFRecHitDeviceCollection> pfRecHitsToken_;
    const edm::EDPutTokenT<cms_uint32_t> sizeToken_;
    const bool synchronise_;

    // data members used to communicate between acquire() and produce()
    cms::alpakatools::host_buffer<uint32_t> size_;
    std::optional<reco::PFRecHitDeviceCollection> pfRecHits_;
  };

  using PFRecHitSoAProducerHCAL = PFRecHitSoAProducer<HCAL>;
  using PFRecHitSoAProducerECAL = PFRecHitSoAProducer<ECAL>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFRecHitSoAProducerHCAL);
DEFINE_FWK_ALPAKA_MODULE(PFRecHitSoAProducerECAL);
