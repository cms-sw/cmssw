
/*----------------------------------------------------------------------

Toy EDProducers of doubles for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
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

  class ToyDoubleProducer : public edm::global::EDProducer<> {
  public:
    explicit ToyDoubleProducer(edm::ParameterSet const& p) : value_(p.getParameter<double>("dvalue")) {
      produces<DoubleProduct>();
    }
    explicit ToyDoubleProducer(double d) : value_(d) { produces<DoubleProduct>(); }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const final;

  private:
    double value_;
  };

  void ToyDoubleProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // Make output
    e.put(std::make_unique<DoubleProduct>(value_));
  }
}  // namespace edmtest

using edmtest::ToyDoubleProducer;
DEFINE_FWK_MODULE(ToyDoubleProducer);
