
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/src/ToyProducts.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {
  // Toy producers
  
  class IntProducer : public edm::EDProducer {
  public:
    explicit IntProducer(edm::ParameterSet const& p) : value_(p.getParameter<int>("ivalue")) {
      produces<IntProduct>();
    }
    explicit IntProducer(int i) : value_(i) {
      produces<IntProduct>();
    }
    virtual ~IntProducer() { }
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
  private:
    int value_;
  };

  void
  IntProducer::produce(edm::Event& e, const edm::EventSetup&) {
    // EventSetup is not used.
    std::auto_ptr<IntProduct> p(new IntProduct(value_));
    e.put(p);
  }
  
  class DoubleProducer : public edm::EDProducer {
  public:
    explicit DoubleProducer(edm::ParameterSet const& p) : value_(p.getParameter<double>("dvalue")) {
      produces<DoubleProduct>();
    }
    explicit DoubleProducer(double d) : value_(d) {
      produces<DoubleProduct>();
    }
    virtual ~DoubleProducer() { }
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
  private:
    double value_;
  };

  void
  DoubleProducer::produce(edm::Event& e, const edm::EventSetup&) {
    // EventSetup is not used.
    // Get input
    edm::Handle<IntProduct> h;
    assert(!h.isValid());

    try {
	std::string emptyLabel;
	e.getByLabel(emptyLabel, h);
	assert ("Failed to throw necessary exception" == 0);
    }
    catch (edm::Exception& x) {
	assert(!h.isValid());
    }
    catch (...) {
	assert("Threw wrong exception" == 0);
    }

    // Make output
    std::auto_ptr<DoubleProduct> p(new DoubleProduct(value_));
    e.put(p);
  }

  class SCSimpleProducer : public edm::EDProducer {
  public:
    explicit SCSimpleProducer(edm::ParameterSet const& p) : size_(p.getParameter<int>("size")) {
      produces<SCSimpleProduct>();
    }

    explicit SCSimpleProducer(int i) : size_(i) {
      produces<SCSimpleProduct>();
    }

    virtual ~SCSimpleProducer() { }
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    
  private:
    int size_;  // number of Simples to put in the collection
  };


  void
  SCSimpleProducer::produce(edm::Event& e, const edm::EventSetup& /* unused */) {
    std::auto_ptr<SCSimpleProduct> p(new SCSimpleProduct(size_));

    SCSimpleProduct::iterator i = p->begin();
    SCSimpleProduct::iterator e = p->end();
    int idx = size_;

    // Fill up the collection so that it is sorted *backwards*.
    while ( i != e )
      {
	i->key = 100 + size_;
	i->value = 1.5 * i->key;
	--idx;	  
	++i;
      }
    
    // Put the product into the Event, thus sorting it.
    e.put(p);
  }
}

using edmtest::IntProducer;
using edmtest::DoubleProducer;
using edmtest::SCSimpleProducer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(IntProducer)
DEFINE_ANOTHER_FWK_MODULE(DoubleProducer)
DEFINE_ANOTHER_FWK_MODULE(SCSimpleProducer)
