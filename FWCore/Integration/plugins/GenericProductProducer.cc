#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingWithDoNotSort.h"
#include "DataFormats/TestObjects/interface/ThingWithPostInsert.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GenericProduct.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  class GenericProductProducer : public edm::global::EDProducer<> {
  public:
    explicit GenericProductProducer(edm::ParameterSet const& ps);

    void produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDPutToken intToken_;
    edm::EDPutToken floatToken_;
    edm::EDPutToken stringToken_;
    edm::EDPutToken vectorToken_;
    edm::EDPutToken thingToken_;
    edm::EDPutToken thingWithPostInsertToken_;
    edm::EDPutToken thingWithDoNotSortToken_;
    edm::EDPutToken badToken_;
  };

  GenericProductProducer::GenericProductProducer(edm::ParameterSet const& iConfig)
      : intToken_(produces(edm::TypeID(typeid(int)))),
        floatToken_(produces(edm::TypeID(typeid(float)))),
        stringToken_(produces(edm::TypeID(typeid(std::string)))),
        vectorToken_(produces(edm::TypeID(typeid(std::vector<double>)))),
        thingToken_(produces(edm::TypeID(typeid(Thing)))),
        thingWithPostInsertToken_(produces(edm::TypeID(typeid(ThingWithPostInsert)))),
        thingWithDoNotSortToken_(produces(edm::TypeID(typeid(ThingWithDoNotSort)))),
        badToken_(produces(edm::TypeID(typeid(int)), "bad")) {}

  void GenericProductProducer::produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const {
    {
      // Check that an int can be put into the event via a GenericProduct.
      int object = 42;
      auto product = std::make_unique<edm::GenericProduct>(object);  // product does not own the underlying object
      assert(product->object().objectCast<int>() == 42);
      auto handle = event.put(intToken_, std::move(product));  // move the underlying object into the event
      assert(handle->objectCast<int>() == 42);
    }

    {
      // Check that a float can be put into the event via a GenericProduct.
      float object = 3.14159f;
      auto product = std::make_unique<edm::GenericProduct>(object);
      assert(product->object().objectCast<float>() == 3.14159f);
      auto handle = event.put(floatToken_, std::move(product));
      assert(handle->objectCast<float>() == 3.14159f);
    }

    {
      // Check that a string can be put into the event via a GenericProduct.
      // With small string optimisation, the underlying buffer should be inside the string object itself.
      std::string object = "42";
      auto product = std::make_unique<edm::GenericProduct>(object);
      assert(product->object().objectCast<std::string>() == "42");
      auto handle = event.put(stringToken_, std::move(product));
      assert(object.empty());  // the original string has been moved
      assert(handle->objectCast<std::string>() == "42");
    }

    {
      // Check that a vector can be put into the event via a GenericProduct.
      // The underlying buffer is outside the vector object itself.
      using Vector = std::vector<double>;
      Vector object = {1., 1., 2., 3., 5., 8., 11., 19., 30.};
      auto product = std::make_unique<edm::GenericProduct>(object);
      assert(product->object().objectCast<Vector>() == Vector({1., 1., 2., 3., 5., 8., 11., 19., 30.}));
      auto handle = event.put(vectorToken_, std::move(product));
      assert(object.empty());  // the original vector has been moved
      assert(handle->objectCast<Vector>() == Vector({1., 1., 2., 3., 5., 8., 11., 19., 30.}));
    }

    {
      // Check that a Thing can be put into the event via a GenericProduct.
      Thing object(99);
      auto product = std::make_unique<edm::GenericProduct>(object);
      assert(product->object().objectCast<Thing>().a == 99);
      auto handle = event.put(thingToken_, std::move(product));
      assert(handle->objectCast<Thing>().a == 99);
    }

    {
      // Check that a ThingWithPostInsert can be put into the event via a GenericProduct,
      // and that ThingWithPostInsert::post_insert() is called by event.put().
      ThingWithPostInsert object(2147483647);
      auto product = std::make_unique<edm::GenericProduct>(object);
      assert(product->object().objectCast<ThingWithPostInsert>().value() == 2147483647);
      assert(not product->object().objectCast<ThingWithPostInsert>().valid());
      auto handle = event.put(thingWithPostInsertToken_, std::move(product));
      assert(handle->objectCast<ThingWithPostInsert>().value() == 2147483647);
      assert(handle->objectCast<ThingWithPostInsert>().valid());
    }

    {
      // Check that a ThingWithDoNotSort can be put into the event via a GenericProduct,
      // and that ThingWithDoNotSort::post_insert() is *not* called by event.put().
      ThingWithDoNotSort object(2147483647);
      auto product = std::make_unique<edm::GenericProduct>(object);
      assert(product->object().objectCast<ThingWithDoNotSort>().value() == 2147483647);
      auto handle = event.put(thingWithDoNotSortToken_, std::move(product));
      assert(handle->objectCast<ThingWithDoNotSort>().value() == 2147483647);
    }

    {
      // Check that mismatches in the type of the token and of the underlying object are caught
      float object = 3.14159f;
      auto product = std::make_unique<edm::GenericProduct>(object);
      assert(product->object().objectCast<float>() == 3.14159f);
      bool failed = false;
      try {
        event.put(badToken_, std::move(product));
      } catch (edm::Exception const& e) {
        assert(e.categoryCode() == edm::errors::LogicError);
        failed = true;
      }
      assert(failed);
    }
  }

  void GenericProductProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
using edmtest::GenericProductProducer;
DEFINE_FWK_MODULE(GenericProductProducer);
