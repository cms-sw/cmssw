// acquire() allocates two separate ThingCollection members; produce()
// extends each further and puts them as two distinct main products: one is
// transformed synchronously (registerTransform), the other asynchronously
// (registerTransformAsync).

#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "AllocProducerTestHelpers.h"

namespace allocMonTest {
  class ExternalWorkTransformAllocProducer : public edm::stream::EDProducer<edm::ExternalWork, edm::Transformer> {
  public:
    explicit ExternalWorkTransformAllocProducer(edm::ParameterSet const&)
        : putTokenSync_(produces<edmtest::ThingCollection>("sync")),
          putTokenAsync_(produces<edmtest::ThingCollection>("async")) {
      registerTransform(
          putTokenSync_,
          [](edm::StreamID, edmtest::ThingCollection const&) { return makeThings(10, 4000); },
          "syncTransformed");
      registerTransformAsync(
          putTokenAsync_,
          [](edm::StreamID, edmtest::ThingCollection const&, edm::WaitingTaskWithArenaHolder holder) {
            return makeThings(10, 5000);
          },
          [](edm::StreamID, edmtest::ThingCollection things) {
            for (int i = 0; i < 10; ++i) {
              things.emplace_back(i + 6000);
            }
            return things;
          },
          "asyncTransformed");
    }

    void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) override {
      syncSourceThings_ = makeThings(10, 7000);
      asyncSourceThings_ = makeThings(10, 8000);
    }

    void produce(edm::Event& event, edm::EventSetup const&) override {
      for (int i = 0; i < 10; ++i) {
        syncSourceThings_.emplace_back(i + 100);
        asyncSourceThings_.emplace_back(i + 200);
      }
      event.emplace(putTokenSync_, std::move(syncSourceThings_));
      event.emplace(putTokenAsync_, std::move(asyncSourceThings_));
    }

  private:
    edm::EDPutTokenT<edmtest::ThingCollection> putTokenSync_;
    edm::EDPutTokenT<edmtest::ThingCollection> putTokenAsync_;
    edmtest::ThingCollection syncSourceThings_;
    edmtest::ThingCollection asyncSourceThings_;
  };
}  // namespace allocMonTest

DEFINE_FWK_MODULE(allocMonTest::ExternalWorkTransformAllocProducer);
