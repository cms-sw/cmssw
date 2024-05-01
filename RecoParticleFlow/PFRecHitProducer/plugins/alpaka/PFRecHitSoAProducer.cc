#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsRecord.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyRecord.h"

#include "CalorimeterDefinitions.h"
#include "PFRecHitProducerKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace particleFlowRecHitProducer;

  template <typename CAL>
  class PFRecHitSoAProducer : public global::EDProducer<> {
  public:
    PFRecHitSoAProducer(edm::ParameterSet const& config)
        : topologyToken_(esConsumes(config.getParameter<edm::ESInputTag>("topology"))),
          pfRecHitsToken_(produces()),
          synchronise_(config.getUntrackedParameter<bool>("synchronise")) {
      // Workaround until the ProductID problem in issue https://github.com/cms-sw/cmssw/issues/44643 is fixed
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
      producesTemporarily("edm::DeviceProduct<alpaka_cuda_async::reco::PFRecHitDeviceCollection>");
#endif

      const std::vector<edm::ParameterSet> producers = config.getParameter<std::vector<edm::ParameterSet>>("producers");
      recHitsToken_.reserve(producers.size());
      for (const edm::ParameterSet& producer : producers) {
        recHitsToken_.emplace_back(consumes(producer.getParameter<edm::InputTag>("src")),
                                   esConsumes(producer.getParameter<edm::ESInputTag>("params")));
      }
    }

    void produce(edm::StreamID, device::Event& event, const device::EventSetup& setup) const override {
      const typename CAL::TopologyTypeDevice& topology = setup.getData(topologyToken_);

      uint32_t num_recHits = 0;
      for (const auto& token : recHitsToken_)
        num_recHits += event.get(token.first)->metadata().size();

      reco::PFRecHitDeviceCollection pfRecHits{(int)num_recHits, event.queue()};

      if (num_recHits != 0) {
        PFRecHitProducerKernel<CAL> kernel{event.queue(), num_recHits};
        for (const auto& token : recHitsToken_)
          kernel.processRecHits(
              event.queue(), event.get(token.first), setup.getData(token.second), topology, pfRecHits);
        kernel.associateTopologyInfo(event.queue(), topology, pfRecHits);
      }

      if (synchronise_)
        alpaka::wait(event.queue());

      event.emplace(pfRecHitsToken_, std::move(pfRecHits));
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
    const device::ESGetToken<typename CAL::TopologyTypeDevice, typename CAL::TopologyRecordType> topologyToken_;
    std::vector<std::pair<device::EDGetToken<typename CAL::CaloRecHitSoATypeDevice>,
                          device::ESGetToken<typename CAL::ParameterType, typename CAL::ParameterRecordType>>>
        recHitsToken_;
    const device::EDPutToken<reco::PFRecHitDeviceCollection> pfRecHitsToken_;
    const bool synchronise_;
  };

  using PFRecHitSoAProducerHCAL = PFRecHitSoAProducer<HCAL>;
  using PFRecHitSoAProducerECAL = PFRecHitSoAProducer<ECAL>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFRecHitSoAProducerHCAL);
DEFINE_FWK_ALPAKA_MODULE(PFRecHitSoAProducerECAL);
