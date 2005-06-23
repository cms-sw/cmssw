#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"
#include "FWCore/FWCoreIntegration/test/HierarchicalEDProducer.h"

namespace edmtest {

  HierarchicalEDProducer::HierarchicalEDProducer(edm::ParameterSet const& ps) :
    radius_ (ps.getParameter<double>("radius")),
    outer_alg_(ps.getParameter<edm::ParameterSet>("nest_1"))
  { }

  // Virtual destructor needed.
  HierarchicalEDProducer::~HierarchicalEDProducer() {}  

  // Functions that gets called by framework every event
  void HierarchicalEDProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // nothing to do ... is just a dummy!
  }

DEFINE_FWK_MODULE(HierarchicalEDProducer)
}
