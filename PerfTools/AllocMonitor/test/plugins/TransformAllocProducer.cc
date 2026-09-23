#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "AllocProducerTestHelpers.h"

namespace allocMonTest {
  class TransformAllocProducer : public edm::global::EDProducer<edm::Transformer> {
  public:
    explicit TransformAllocProducer(edm::ParameterSet const&)
        : putTokenOne_(produces<edmtest::ThingCollection>("one")),
          putTokenTwo_(produces<edmtest::ThingCollection>("two")),
          putTokenThree_(produces<edmtest::ThingCollection>("three")) {
      registerTransform(
          putTokenOne_,
          [](edm::StreamID, edmtest::ThingCollection const&) { return makeThings(10, 1000); },
          "oneTransformed");
      registerTransform(
          putTokenTwo_,
          [](edm::StreamID, edmtest::ThingCollection const&) { return makeThings(10, 2000); },
          "twoTransformed");
      registerTransform(
          putTokenThree_,
          [](edm::StreamID, edmtest::ThingCollection const&) { return makeThings(10, 3000); },
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

DEFINE_FWK_MODULE(allocMonTest::TransformAllocProducer);
