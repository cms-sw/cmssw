#include "DataFormats/FWLite/test/VectorIntProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edmtest {
  VectorIntProducer::VectorIntProducer(edm::ParameterSet const& iConfig)
  {
    produces<std::vector<int> >();
  }

  // Virtual destructor needed.
  VectorIntProducer::~VectorIntProducer() { }  

  // Functions that gets called by framework every event
  void VectorIntProducer::produce(edm::Event& e, edm::EventSetup const&) {
    std::auto_ptr<std::vector<int> > result(new std::vector<int>);
    result->push_back(42);
    e.put(result);
  }

}
using edmtest::VectorIntProducer;
DEFINE_FWK_MODULE(VectorIntProducer);
