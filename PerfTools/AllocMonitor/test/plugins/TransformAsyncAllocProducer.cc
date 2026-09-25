#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "AllocProducerTestHelpers.h"

namespace allocMonTest {
  class TransformAsyncAllocProducer : public edm::global::EDProducer<edm::Transformer> {
  public:
    explicit TransformAsyncAllocProducer(edm::ParameterSet const&)
        : putTokenOne_(produces<edmtest::ThingCollection>("one")),
          putTokenTwo_(produces<edmtest::ThingCollection>("two")),
          putTokenThree_(produces<edmtest::ThingCollection>("three")) {
      registerTransformAsync(
          putTokenOne_,
          [](edm::StreamID, edmtest::ThingCollection const&, edm::WaitingTaskWithArenaHolder holder) {
            // Real allocation done synchronously; the holder parameter is
            // simply let go out of scope (dropped) at the end, which
            // completes the acquiring phase automatically.
            return makeThings(10, 4000);
          },
          [](edm::StreamID, edmtest::ThingCollection things) {
            for (int i = 0; i < 10; ++i) {
              things.emplace_back(i + 5000);
            }
            return things;
          },
          "oneTransformed");
      registerTransformAsync(
          putTokenTwo_,
          [](edm::StreamID, edmtest::ThingCollection const&, edm::WaitingTaskWithArenaHolder holder) {
            return makeThings(10, 6000);
          },
          [](edm::StreamID, edmtest::ThingCollection things) {
            for (int i = 0; i < 10; ++i) {
              things.emplace_back(i + 7000);
            }
            return things;
          },
          "twoTransformed");
      registerTransformAsync(
          putTokenThree_,
          [](edm::StreamID, edmtest::ThingCollection const&, edm::WaitingTaskWithArenaHolder holder) {
            return makeThings(10, 8000);
          },
          [](edm::StreamID, edmtest::ThingCollection things) {
            for (int i = 0; i < 10; ++i) {
              things.emplace_back(i + 9000);
            }
            return things;
          },
          "threeTransformed");
    }

    void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
      event.emplace(putTokenOne_, makeThings(10, 0));
      event.emplace(putTokenTwo_, makeThings(10, 100));
      event.emplace(putTokenThree_, makeThings(10, 200));
    }

  private:
    edm::EDPutTokenT<edmtest::ThingCollection> putTokenOne_;
    edm::EDPutTokenT<edmtest::ThingCollection> putTokenTwo_;
    edm::EDPutTokenT<edmtest::ThingCollection> putTokenThree_;
  };
}  // namespace allocMonTest

DEFINE_FWK_MODULE(allocMonTest::TransformAsyncAllocProducer);
