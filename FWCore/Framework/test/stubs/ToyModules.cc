
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
      produces<int>();
    }
    explicit IntProducer(int i) : value_(i) {
      produces<int>();
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
      produces<double>();
    }
    explicit DoubleProducer(double d) : value_(d) {
      produces<double>();
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
}
using edmtest::IntProducer;
using edmtest::DoubleProducer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(IntProducer)
DEFINE_ANOTHER_FWK_MODULE(DoubleProducer)
