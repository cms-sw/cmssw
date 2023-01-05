#include "FWCore/Framework/interface/MakerMacros.h"
#include "HierarchicalEDProducer.h"

namespace edmtest {

  HierarchicalEDProducer::HierarchicalEDProducer(edm::ParameterSet const& ps)
      : radius_(ps.getParameter<double>("radius")), outer_alg_(ps.getParameterSet("nest_1")) {
    produces<int>();
  }

  // Virtual destructor needed.
  HierarchicalEDProducer::~HierarchicalEDProducer() {}

  // Functions that gets called by framework every event
  void HierarchicalEDProducer::produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const {
    // nothing to do ... is just a dummy!
  }
}  // namespace edmtest
using edmtest::HierarchicalEDProducer;
DEFINE_FWK_MODULE(HierarchicalEDProducer);
