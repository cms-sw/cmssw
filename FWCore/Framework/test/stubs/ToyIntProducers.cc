
/*----------------------------------------------------------------------

Toy EDProducers of Ints for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
//
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
//
#include <cassert>
#include <string>
#include <vector>

namespace edmtest {

  //--------------------------------------------------------------------
  //
  // Int producers
  //
  //--------------------------------------------------------------------

  //--------------------------------------------------------------------
  //
  // throws an exception.
  // Announces an IntProduct but does not produce one;
  // every call to FailingProducer::produce throws a cms exception
  //
  class FailingProducer : public edm::EDProducer {
  public:
    explicit FailingProducer(edm::ParameterSet const& /*p*/) {
      produces<IntProduct>();
    }
    virtual ~FailingProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  };

  void
  FailingProducer::produce(edm::Event&, edm::EventSetup const&) {
    // We throw an edm exception with a configurable action.
    throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
  }

  //--------------------------------------------------------------------
  //
  // Announces an IntProduct but does not produce one;
  // every call to NonProducer::produce does nothing.
  //
  class NonProducer : public edm::EDProducer {
  public:
    explicit NonProducer(edm::ParameterSet const& /*p*/) {
      produces<IntProduct>();
    }
    virtual ~NonProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  };

  void
  NonProducer::produce(edm::Event&, edm::EventSetup const&) {
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance.
  //
  // NOTE: this really should be a global::EDProducer<> but for testing we use stream
  class IntProducer : public edm::stream::EDProducer<> {
  public:
    explicit IntProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<int>("ivalue")) {
      produces<IntProduct>();
    }
    explicit IntProducer(int i) : value_(i) {
      produces<IntProduct>();
    }
    virtual ~IntProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    int value_;
  };

  //--------------------------------------------------------------------

  void
  IntProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::unique_ptr<IntProduct> p(new IntProduct(value_));
    e.put(std::move(p));
  }

  class ConsumingIntProducer : public edm::stream::EDProducer<> {
  public:
    explicit ConsumingIntProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<int>("ivalue")) {
      produces<IntProduct>();
      // not used, only exists to test PathAndConsumesOfModules
      consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"));
      consumesMany<edm::TriggerResults>();
    }
    explicit ConsumingIntProducer(int i) : value_(i) {
      produces<IntProduct>();
      // not used, only exists to test PathAndConsumesOfModules
      consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"));
      consumesMany<edm::TriggerResults>();
    }
    virtual ~ConsumingIntProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    int value_;
  };

  void
  ConsumingIntProducer::produce(edm::Event& e, edm::EventSetup const&) {
    std::unique_ptr<IntProduct> p(new IntProduct(value_));
    e.put(std::move(p));
  }

  //--------------------------------------------------------------------
  //
  // Produces an IntProduct instance whose value is the event number,
  // rather than the value of a configured parameter.
  //
  class EventNumberIntProducer : public edm::EDProducer {
  public:
    explicit EventNumberIntProducer(edm::ParameterSet const&) {
      produces<UInt64Product>();
    }
    EventNumberIntProducer() {
      produces<UInt64Product>();
    }
    virtual ~EventNumberIntProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
  };

  void
  EventNumberIntProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::unique_ptr<UInt64Product> p(new UInt64Product(e.id().event()));
    e.put(std::move(p));
  }


  //--------------------------------------------------------------------
  //
  // Produces a TransientIntProduct instance.
  //
  class TransientIntProducer : public edm::EDProducer {
  public:
    explicit TransientIntProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<int>("ivalue")) {
      produces<TransientIntProduct>();
    }
    explicit TransientIntProducer(int i) : value_(i) {
      produces<TransientIntProduct>();
    }
    virtual ~TransientIntProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    int value_;
  };

  void
  TransientIntProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::unique_ptr<TransientIntProduct> p(new TransientIntProduct(value_));
    e.put(std::move(p));
  }

  //--------------------------------------------------------------------
  //
  // Produces a IntProduct instance from a TransientIntProduct
  //
  class IntProducerFromTransient : public edm::EDProducer {
  public:
    explicit IntProducerFromTransient(edm::ParameterSet const&) {
      produces<IntProduct>();
      consumes<TransientIntProduct>(edm::InputTag{"TransientThing"});
    }
    explicit IntProducerFromTransient() {
      produces<IntProduct>();
    }
    virtual ~IntProducerFromTransient() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
  };

  void
  IntProducerFromTransient::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    edm::Handle<TransientIntProduct> result;
    bool ok = e.getByLabel("TransientThing", result);
    assert(ok);
    std::unique_ptr<IntProduct> p(new IntProduct(result.product()->value));
    e.put(std::move(p));
  }

  //--------------------------------------------------------------------
  //
  // Produces an Int16_tProduct instance.
  //
  class Int16_tProducer : public edm::EDProducer {
  public:
    explicit Int16_tProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<int>("ivalue")) {
      produces<Int16_tProduct>();
    }
    explicit Int16_tProducer(boost::int16_t i, boost::uint16_t) : value_(i) {
      produces<Int16_tProduct>();
    }
    virtual ~Int16_tProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    boost::int16_t value_;
  };

  void
  Int16_tProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    std::unique_ptr<Int16_tProduct> p(new Int16_tProduct(value_));
    e.put(std::move(p));
  }

  //
  // Produces an IntProduct instance, using an IntProduct as input.
  //

  class AddIntsProducer : public edm::EDProducer {
  public:
    explicit AddIntsProducer(edm::ParameterSet const& p) :
        labels_(p.getParameter<std::vector<std::string> >("labels")) {
      produces<IntProduct>();
      for( auto const& label: labels_) {
        consumes<IntProduct>(edm::InputTag{label});
      }
    }
    virtual ~AddIntsProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  private:
    std::vector<std::string> labels_;
  };

  void
  AddIntsProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    int value = 0;
    for(std::vector<std::string>::iterator itLabel = labels_.begin(), itLabelEnd = labels_.end();
        itLabel != itLabelEnd; ++itLabel) {
      edm::Handle<IntProduct> anInt;
      e.getByLabel(*itLabel, anInt);
      value += anInt->value;
    }
    std::unique_ptr<IntProduct> p(new IntProduct(value));
    e.put(std::move(p));
  }

}

using edmtest::FailingProducer;
using edmtest::NonProducer;
using edmtest::IntProducer;
using edmtest::ConsumingIntProducer;
using edmtest::EventNumberIntProducer;
using edmtest::TransientIntProducer;
using edmtest::IntProducerFromTransient;
using edmtest::Int16_tProducer;
using edmtest::AddIntsProducer;
DEFINE_FWK_MODULE(FailingProducer);
DEFINE_FWK_MODULE(NonProducer);
DEFINE_FWK_MODULE(IntProducer);
DEFINE_FWK_MODULE(ConsumingIntProducer);
DEFINE_FWK_MODULE(EventNumberIntProducer);
DEFINE_FWK_MODULE(TransientIntProducer);
DEFINE_FWK_MODULE(IntProducerFromTransient);
DEFINE_FWK_MODULE(Int16_tProducer);
DEFINE_FWK_MODULE(AddIntsProducer);
