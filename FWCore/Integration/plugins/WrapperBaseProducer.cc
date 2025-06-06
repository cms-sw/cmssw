#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingWithDoNotSort.h"
#include "DataFormats/TestObjects/interface/ThingWithPostInsert.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

// FIXME for do_post_insert_if_available, remove when no longer needed
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"

namespace edmtest {

  class WrapperBaseProducer : public edm::global::EDProducer<> {
  public:
    explicit WrapperBaseProducer(edm::ParameterSet const& ps);

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

  template <typename T>
  static T& unwrap_as(edm::WrapperBase& wrapper) {
    // throws std::bad_cast on conversion error
    edm::Wrapper<T>& w = dynamic_cast<edm::Wrapper<T>&>(wrapper);

    // throws an execption if the wrapper is empty
    if (w.product() == nullptr) {
      throw cms::Exception("LogicError") << "Empty or invalid wrapper";
    }

    // return the content of the wrapper
    return w.bareProduct();
  }

  template <typename T>
  static T const& unwrap_as(edm::WrapperBase const& wrapper) {
    // throws std::bad_cast on conversion error
    edm::Wrapper<T> const& w = dynamic_cast<edm::Wrapper<T> const&>(wrapper);

    // throws an execption if the wrapper is empty
    if (w.product() == nullptr) {
      throw cms::Exception("LogicError") << "Empty or invalid wrapper";
    }

    // return the content of the wrapper
    return *w.product();
  }

  WrapperBaseProducer::WrapperBaseProducer(edm::ParameterSet const& iConfig)
      : intToken_(produces(edm::TypeID(typeid(int)))),
        floatToken_(produces(edm::TypeID(typeid(float)))),
        stringToken_(produces(edm::TypeID(typeid(std::string)))),
        vectorToken_(produces(edm::TypeID(typeid(std::vector<double>)))),
        thingToken_(produces(edm::TypeID(typeid(Thing)))),
        thingWithPostInsertToken_(produces(edm::TypeID(typeid(ThingWithPostInsert)))),
        thingWithDoNotSortToken_(produces(edm::TypeID(typeid(ThingWithDoNotSort)))),
        badToken_(produces(edm::TypeID(typeid(int)), "bad")) {}

  void WrapperBaseProducer::produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const {
    {
      // Check that an int can be put into the event via a WrapperBase.
      int value = 42;
      // Copy the value into the wrapper
      std::unique_ptr<edm::WrapperBase> product(new edm::Wrapper<int>(edm::WrapperBase::Emplace{}, value));
      // FIXME Wrapper should call post_insert (if available), but that is not implemented yet
      edm::detail::do_post_insert_if_available(unwrap_as<int>(*product));
      assert(unwrap_as<int>(*product) == value);
      // Move the wrapper into the event
      auto handle = event.put(intToken_, std::move(product));
      assert(not product);
      assert(handle.isValid());
      assert(unwrap_as<int>(*handle) == value);
    }

    {
      // Check that a float can be put into the event via a WrapperBase.
      float value = 3.14159f;
      // Copy the value into the wrapper
      std::unique_ptr<edm::WrapperBase> product(new edm::Wrapper<float>(edm::WrapperBase::Emplace{}, value));
      // FIXME Wrapper should call post_insert (if available), but that is not implemented yet
      edm::detail::do_post_insert_if_available(unwrap_as<float>(*product));
      assert(unwrap_as<float>(*product) == value);
      // Move the wrapper into the event
      auto handle = event.put(floatToken_, std::move(product));
      assert(not product);
      assert(handle.isValid());
      assert(unwrap_as<float>(*handle) == value);
    }

    {
      // Check that a string can be put into the event via a WrapperBase.
      // With small string optimisation, the underlying buffer should be inside the string object itself.
      std::string value = "42";
      // Copy the value into the wrapper
      std::unique_ptr<edm::WrapperBase> product(new edm::Wrapper<std::string>(edm::WrapperBase::Emplace{}, value));
      // FIXME Wrapper should call post_insert (if available), but that is not implemented yet
      edm::detail::do_post_insert_if_available(unwrap_as<std::string>(*product));
      assert(unwrap_as<std::string>(*product) == value);
      // Move the wrapper into the event
      auto handle = event.put(stringToken_, std::move(product));
      assert(not product);
      assert(handle.isValid());
      assert(unwrap_as<std::string>(*handle) == value);
    }

    {
      // Check that a vector can be put into the event via a WrapperBase.
      // The underlying buffer is outside the vector object itself.
      using Vector = std::vector<double>;
      Vector value = {1., 1., 2., 3., 5., 8., 11., 19., 30.};
      // Copy the value into the wrapper
      std::unique_ptr<edm::WrapperBase> product(new edm::Wrapper<Vector>(edm::WrapperBase::Emplace{}, value));
      // FIXME Wrapper should call post_insert (if available), but that is not implemented yet
      edm::detail::do_post_insert_if_available(unwrap_as<Vector>(*product));
      assert(unwrap_as<Vector>(*product) == value);
      // Move the wrapper into the event
      auto handle = event.put(vectorToken_, std::move(product));
      assert(not product);
      assert(handle.isValid());
      assert(unwrap_as<Vector>(*handle) == value);
    }

    {
      // Check that a Thing can be put into the event via a WrapperBase.
      cms_int32_t value = 99;
      // Copy the value into the wrapper
      std::unique_ptr<edm::WrapperBase> product(new edm::Wrapper<Thing>(edm::WrapperBase::Emplace{}, value));
      // FIXME Wrapper should call post_insert (if available), but that is not implemented yet
      edm::detail::do_post_insert_if_available(unwrap_as<Thing>(*product));
      assert(unwrap_as<Thing>(*product).a == value);
      // Move the wrapper into the event
      auto handle = event.put(thingToken_, std::move(product));
      assert(not product);
      assert(handle.isValid());
      assert(unwrap_as<Thing>(*handle).a == value);
    }

    {
      // Check that a ThingWithPostInsert can be put into the event via a WrapperBase,
      // and that ThingWithPostInsert::post_insert() is called by Wrapper.
      int32_t value = 2147483647;
      // Copy the value into the wrapper
      std::unique_ptr<edm::WrapperBase> product(
          new edm::Wrapper<ThingWithPostInsert>(edm::WrapperBase::Emplace{}, value));
      // FIXME Wrapper should call post_insert (if available), but that is not implemented yet
      assert(not unwrap_as<ThingWithPostInsert>(*product).valid());
      edm::detail::do_post_insert_if_available(unwrap_as<ThingWithPostInsert>(*product));
      // end-of-FIXME
      assert(unwrap_as<ThingWithPostInsert>(*product).value() == value);
      assert(unwrap_as<ThingWithPostInsert>(*product).valid());
      // Move the wrapper into the event
      auto handle = event.put(thingWithPostInsertToken_, std::move(product));
      assert(not product);
      assert(handle.isValid());
      assert(unwrap_as<ThingWithPostInsert>(*handle).value() == value);
      assert(unwrap_as<ThingWithPostInsert>(*handle).valid());
    }

    {
      // Check that a ThingWithDoNotSort can be put into the event via a WrapperBase,
      // and that ThingWithDoNotSort::post_insert() is *not* called by Wrapper.
      int32_t value = 2147483647;
      // Copy the value into the wrapper
      std::unique_ptr<edm::WrapperBase> product(
          new edm::Wrapper<ThingWithDoNotSort>(edm::WrapperBase::Emplace{}, value));
      // FIXME Wrapper should call post_insert (if available), but that is not implemented yet
      edm::detail::do_post_insert_if_available(unwrap_as<ThingWithDoNotSort>(*product));
      assert(unwrap_as<ThingWithDoNotSort>(*product).value() == value);
      // Move the wrapper into the event
      auto handle = event.put(thingWithDoNotSortToken_, std::move(product));
      assert(not product);
      assert(handle.isValid());
      assert(unwrap_as<ThingWithDoNotSort>(*handle).value() == value);
    }

    {
      // Check that mismatches in the type of the token and of the underlying object are caught
      float value = 3.14159f;
      // Copy the value into the wrapper
      std::unique_ptr<edm::WrapperBase> product(new edm::Wrapper<float>(edm::WrapperBase::Emplace{}, value));
      // FIXME Wrapper should call post_insert (if available), but that is not implemented yet
      edm::detail::do_post_insert_if_available(unwrap_as<float>(*product));
      assert(unwrap_as<float>(*product) == value);
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

  void WrapperBaseProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
using edmtest::WrapperBaseProducer;
DEFINE_FWK_MODULE(WrapperBaseProducer);
