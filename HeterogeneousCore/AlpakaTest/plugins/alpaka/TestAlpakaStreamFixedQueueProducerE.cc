#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a stream FixedQueueEDProducer that
   * - consumes a device ESProduct
   * - consumes a device EDProduct
   * - produces a device EDProduct (that can get transferred to host automatically)
   *
   * Unlike stream::EDProducer, that uses a random device queue for every event,
   * stream::FixedQueueEDProducer always uses the same device queue for all the events
   * processed by a given framework stream.
   *
   * This module tests that the queue being used is always the same, even if a device
   * product with a reusable queue is consumed first.
   */
  class TestAlpakaStreamFixedQueueProducerE : public stream::FixedQueueEDProducer<> {
  public:
    TestAlpakaStreamFixedQueueProducerE(edm::ParameterSet const& config)
        : FixedQueueEDProducer<>(config),
          esToken_(esConsumes(config.getParameter<edm::ESInputTag>("eventSetupSource"))),
          getToken_(consumes(config.getParameter<edm::InputTag>("source"))),
          getTokenMulti2_(consumes(config.getParameter<edm::InputTag>("source"))),
          getTokenMulti3_(consumes(config.getParameter<edm::InputTag>("source"))),
          putToken_{produces()},
          putTokenMulti2_{produces()},
          putTokenMulti3_{produces()} {}

    void beginStream(edm::StreamID sid, Queue queue) override { queue_ = queue; }

    void endStream(Queue queue) override { queue_.reset(); }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto const& esData = iSetup.getData(esToken_);
      auto const& input = iEvent.get(getToken_);
      auto const& inputMulti2 = iEvent.get(getTokenMulti2_);
      auto const& inputMulti3 = iEvent.get(getTokenMulti3_);

      // run the algorithm, potentially asynchronously
      auto deviceProduct = algo_.update(iEvent.queue(), input, esData);
      auto deviceProductMulti2 = algo_.updateMulti2(iEvent.queue(), inputMulti2, esData);
      auto deviceProductMulti3 = algo_.updateMulti3(iEvent.queue(), inputMulti3, esData);

      iEvent.emplace(putToken_, std::move(deviceProduct));
      iEvent.emplace(putTokenMulti2_, std::move(deviceProductMulti2));
      iEvent.emplace(putTokenMulti3_, std::move(deviceProductMulti3));

      assert(iEvent.queue() == *queue_);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("eventSetupSource", edm::ESInputTag{});
      desc.add("source", edm::InputTag{});

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<AlpakaESTestDataEDevice, AlpakaESTestRecordC> esToken_;
    const device::EDGetToken<portabletest::TestDeviceCollection> getToken_;
    const device::EDGetToken<portabletest::TestDeviceCollection2> getTokenMulti2_;
    const device::EDGetToken<portabletest::TestDeviceCollection3> getTokenMulti3_;
    const device::EDPutToken<portabletest::TestDeviceCollection> putToken_;
    const device::EDPutToken<portabletest::TestDeviceCollection2> putTokenMulti2_;
    const device::EDPutToken<portabletest::TestDeviceCollection3> putTokenMulti3_;

    // implementation of the algorithm
    TestAlgo algo_;

    // validation of the queue logic
    std::optional<Queue> queue_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaStreamFixedQueueProducerE);
