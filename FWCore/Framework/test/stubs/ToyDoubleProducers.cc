
/*----------------------------------------------------------------------

Toy EDProducers of doubles for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>

namespace edmtest {

  //--------------------------------------------------------------------
  //
  // Toy double producers
  //
  //--------------------------------------------------------------------

  //--------------------------------------------------------------------
  //
  // Produces an DoubleProduct instance.
  //

  class ToyDoubleProducer : public edm::EDProducer {
  public:
    explicit ToyDoubleProducer(edm::ParameterSet const& p) :
      value_(p.getParameter<double>("dvalue")) {
      produces<DoubleProduct>();
    }
    explicit ToyDoubleProducer(double d) : value_(d) {
      produces<DoubleProduct>();
    }
    virtual ~ToyDoubleProducer() {}
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
  private:
    double value_;
  };

  void
  ToyDoubleProducer::produce(edm::Event& e, edm::EventSetup const&) {

    // Make output
    std::unique_ptr<DoubleProduct> p(new DoubleProduct(value_));
    e.put(std::move(p));
  }

}

using edmtest::ToyDoubleProducer;
DEFINE_FWK_MODULE(ToyDoubleProducer);
