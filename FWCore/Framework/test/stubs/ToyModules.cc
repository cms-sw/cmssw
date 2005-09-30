
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

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
      assert ( size_ > 1 );
    }

    explicit SCSimpleProducer(int i) : size_(i) {
      produces<SCSimpleProduct>();
      assert ( size_ > 1 );
    }

    virtual ~SCSimpleProducer() { }
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    
  private:
    int size_;  // number of Simples to put in the collection
  };


  void
  SCSimpleProducer::produce(edm::Event& e, const edm::EventSetup& /* unused */) {

    // Fill up a collection so that it is sorted *backwards*.
    std::vector<Simple> guts(size_);
    for (int i = 0; i < size_; ++i)
      {
	guts[i].key = size_ - i;
	guts[i].value = 1.5 * i;
      }

    // Verify that the vector is not sorted -- in fact, it is sorted backwards!
    for (int i = 1; i < size_; ++i)
      {
	assert( guts[i-1].id() > guts[i].id());
      }

    std::auto_ptr<SCSimpleProduct> p(new SCSimpleProduct(guts));
    
    // Put the product into the Event, thus sorting it.
    e.put(p);

    // Get the product back out; it should be sorted.
    edm::Handle<SCSimpleProduct> h;
    e.getByType(h);
    assert( h.isValid() );

    // Check the sorting. DO NOT DO THIS IN NORMAL CODE; we are
    // copying all the values out of the SortedCollection so we can
    // manipulate them via an interface different from
    // SortedCollection, just so that we can make sure the collection
    // is sorted.
    std::vector<Simple> after( h->begin(), h->end() );

    // Verify that the vector is not sorted.
    for (int i = 1; i < size_; ++i)
      {
	assert( after[i-1].id() < after[i].id());
      }    
  }
}

using edmtest::IntProducer;
using edmtest::DoubleProducer;
using edmtest::SCSimpleProducer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(IntProducer)
DEFINE_ANOTHER_FWK_MODULE(DoubleProducer)
DEFINE_ANOTHER_FWK_MODULE(SCSimpleProducer)
