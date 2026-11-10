#include <cassert>
#include <memory>
#include <optional>

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingFixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a stream SynchronizingFixedQueueEDProducer that
   * - produces a device EDProduct (that can get transferred to host automatically)
   * - synchronizes in a non-blocking way with the ExternalWork module
   *   ability (via the SynchronizingFixedQueueEDProducer base class)
   *
   * Unlike stream::SynchronizingEDProducer, that uses a random device queue for every event,
   * stream::SynchronizingFixedQueueEDProducer always uses the same device queue for all the
   * events processed by a given framework stream.
   *
   * This module tests that the queue being used is always the same, in both acquire() and
   * produce(), when no device products are consumed.
   */
  class TestAlpakaStreamSynchronizingFixedQueueProducerToDevice : public stream::SynchronizingFixedQueueEDProducer<> {
  public:
    TestAlpakaStreamSynchronizingFixedQueueProducerToDevice(edm::ParameterSet const& iConfig)
        : SynchronizingFixedQueueEDProducer<>(iConfig),
          putToken_{produces()},
          size_{iConfig.getParameter<edm::ParameterSet>("size").getParameter<int32_t>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))} {}

    void beginStream(edm::StreamID sid, Queue queue) override { queue_ = queue; }

    void endStream(Queue queue) override { queue_.reset(); }

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      deviceProduct_ = std::make_unique<portabletest::TestDeviceCollection>(iEvent.queue(), size_);

      // run the algorithm, potentially asynchronously
      algo_.fill(iEvent.queue(), *deviceProduct_);

      assert(iEvent.queue() == *queue_);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      iEvent.put(putToken_, std::move(deviceProduct_));

      assert(iEvent.queue() == *queue_);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;

      edm::ParameterSetDescription psetSize;
      psetSize.add<int32_t>("alpaka_serial_sync");
      psetSize.add<int32_t>("alpaka_cuda_async");
      psetSize.add<int32_t>("alpaka_rocm_async");
      desc.add("size", psetSize);

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<portabletest::TestDeviceCollection> putToken_;
    const int32_t size_;

    // implementation of the algorithm
    TestAlgo algo_;

    std::unique_ptr<portabletest::TestDeviceCollection> deviceProduct_;

    // validation of the queue logic
    std::optional<Queue> queue_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaStreamSynchronizingFixedQueueProducerToDevice);
