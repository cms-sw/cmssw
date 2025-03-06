#include "DataFormats/TestObjects/interface/ThingWithDoNotSort.h"
#include "DataFormats/TestObjects/interface/ThingWithPostInsert.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  class PostInsertProducer : public edm::global::EDProducer<> {
  public:
    explicit PostInsertProducer(edm::ParameterSet const& ps);

    void produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDPutTokenT<ThingWithPostInsert> putTokenPostInsert_;
    edm::EDPutTokenT<ThingWithDoNotSort> putTokenDoNotSort_;
    edm::EDPutTokenT<ThingWithPostInsert> emplaceTokenPostInsert_;
    edm::EDPutTokenT<ThingWithDoNotSort> emplaceTokenDoNotSort_;
  };

  PostInsertProducer::PostInsertProducer(edm::ParameterSet const& iConfig)
      : putTokenPostInsert_{produces<ThingWithPostInsert>("put")},
        putTokenDoNotSort_{produces<ThingWithDoNotSort>("put")},
        emplaceTokenPostInsert_{produces<ThingWithPostInsert>("emplace")},
        emplaceTokenDoNotSort_{produces<ThingWithDoNotSort>("emplace")} {}

  // Functions that gets called by framework every event
  void PostInsertProducer::produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const {
    {
      // Check that event.put() calls ThingWithPostInsert::post_insert().
      auto product = std::make_unique<ThingWithPostInsert>(42);
      assert(not product->valid());
      assert(product->value() == 42);
      auto handle = event.put(putTokenPostInsert_, std::move(product));
      assert(handle->valid());
      assert(handle->value() == 42);
    }

    {
      // Check that event.put *does not* call ThingWithDoNotSort::post_insert(),
      // which would throw an exception.
      auto product = std::make_unique<ThingWithDoNotSort>(42);
      assert(product->value() == 42);
      auto handle = event.put(putTokenDoNotSort_, std::move(product));
      assert(handle->value() == 42);
    }

    {
      // Check that event.emplace() calls ThingWithPostInsert::post_insert().
      ThingWithPostInsert product{42};
      assert(not product.valid());
      assert(product.value() == 42);
      auto handle = event.emplace(emplaceTokenPostInsert_, product);
      assert(handle->valid());
      assert(handle->value() == 42);
    }

    {
      // Check that event.emplace *does not* call ThingWithDoNotSort::post_insert(),
      // which would throw an exception.
      ThingWithDoNotSort product{42};
      assert(product.value() == 42);
      auto handle = event.emplace(emplaceTokenDoNotSort_, product);
      assert(handle->value() == 42);
    }
  }

  void PostInsertProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
using edmtest::PostInsertProducer;
DEFINE_FWK_MODULE(PostInsertProducer);
