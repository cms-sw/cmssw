#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "AllocProducerTestHelpers.h"

namespace allocMonTest {
  class ExternalWorkAllocProducer : public edm::stream::EDProducer<edm::ExternalWork> {
  public:
    explicit ExternalWorkAllocProducer(edm::ParameterSet const&) : putToken_(produces<edmtest::ThingCollection>()) {}

    void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) override {
      acquiredThings_ = makeThings(10, 0);
    }

    void produce(edm::Event& event, edm::EventSetup const&) override {
      // Extend the ThingCollection built in acquire() with more real
      // allocations before putting it into the event.
      for (int i = 0; i < 10; ++i) {
        acquiredThings_.emplace_back(i + 100);
      }
      event.emplace(putToken_, std::move(acquiredThings_));
    }

  private:
    edm::EDPutTokenT<edmtest::ThingCollection> putToken_;
    edmtest::ThingCollection acquiredThings_;
  };
}  // namespace allocMonTest

DEFINE_FWK_MODULE(allocMonTest::ExternalWorkAllocProducer);
