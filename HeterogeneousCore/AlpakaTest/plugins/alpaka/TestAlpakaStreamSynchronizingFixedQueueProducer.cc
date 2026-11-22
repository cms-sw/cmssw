#include <cassert>
#include <optional>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingFixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/TestHostOnlyHelperClass.h"

#include "TestHelperClass.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a stream SynchronizingFixedQueueEDProducer that
   * - uses a helper class (need to use edm::ConsumesCollector), that
   *   - consumes a device EDProduct
   *   - consumes a host ESProduct
   *   - consumes a device ESProduct
   * - uses a non-Alpaka-aware helper class (need to use edm::ConsumesCollector), that
   *   - consumes a host EDProduct
   *   - consumes a host ESProduct
   * - consumes a device ESProduct
   * - produces a host EDProduct
   * - synchronizes in a non-blocking way with the ExternalWork module
   *   ability (via the SynchronizingFixedQueueEDProducer base class)
   *
   * Unlike stream::SynchronizingEDProducer, that uses a random device queue for every event,
   * stream::SynchronizingFixedQueueEDProducer always uses the same device queue for all the
   * events processed by a given framework stream.
   *
   * This module tests that the queue being used is always the same, in both acquire() and
   * produce(), even if a device product with a reusable queue is consumed first.
   */
  class TestAlpakaStreamSynchronizingFixedQueueProducer : public stream::SynchronizingFixedQueueEDProducer<> {
  public:
    TestAlpakaStreamSynchronizingFixedQueueProducer(edm::ParameterSet const& iConfig)
        : SynchronizingFixedQueueEDProducer<>(iConfig),
          esTokenDevice_(esConsumes()),
          putToken_{produces()},
          putTokenMulti2_{produces()},
          putTokenMulti3_{produces()},
          helper_{iConfig, consumesCollector()},
          hostHelper_{iConfig, consumesCollector()},
          expectedInt_{iConfig.getParameter<int>("expectedInt")} {}

    void beginStream(edm::StreamID sid, Queue queue) override { queue_ = queue; }

    void endStream(Queue queue) override { queue_.reset(); }

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      [[maybe_unused]] auto const& esData = iSetup.getData(esTokenDevice_);

      int const value = hostHelper_.run(iEvent, iSetup);
      if (value != expectedInt_) {
        throw cms::Exception("Assert") << "Expected value " << expectedInt_ << ", but got " << value;
      }

      helper_.makeAsync(iEvent, iSetup);

      assert(iEvent.queue() == *queue_);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      iEvent.emplace(putToken_, helper_.moveFrom());
      iEvent.emplace(putTokenMulti2_, helper_.moveFromMulti2());
      iEvent.emplace(putTokenMulti3_, helper_.moveFromMulti3());

      assert(iEvent.queue() == *queue_);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      TestHelperClass::fillPSetDescription(desc);
      cms::alpakatest::TestHostOnlyHelperClass::fillPSetDescription(desc);
      desc.add<int>("expectedInt");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<AlpakaESTestDataDDevice, AlpakaESTestRecordD> esTokenDevice_;
    const edm::EDPutTokenT<portabletest::TestHostCollection> putToken_;
    const edm::EDPutTokenT<portabletest::TestHostCollection2> putTokenMulti2_;
    const edm::EDPutTokenT<portabletest::TestHostCollection3> putTokenMulti3_;

    TestHelperClass helper_;
    cms::alpakatest::TestHostOnlyHelperClass const hostHelper_;
    int const expectedInt_;

    // validation of the queue logic
    std::optional<Queue> queue_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaStreamSynchronizingFixedQueueProducer);
