#include "TH1F.h"

#include "FWCore/Integration/test/HistProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edmtest {
  HistProducer::HistProducer(edm::ParameterSet const&) {
    produces<TH1F>();
    //produces<ThingWithHist>();
  }

  // Virtual destructor needed.
  HistProducer::~HistProducer() { }  

  // Functions that gets called by framework every event
  void HistProducer::produce(edm::Event& e, edm::EventSetup const&) {

    std::auto_ptr<TH1F> result(new TH1F);  //Empty
    e.put(result);
    //std::auto_ptr<ThingWithHist> result2(new ThingWithHist);  //Empty
    //e.put(result2);
  }

}
using edmtest::HistProducer;
DEFINE_FWK_MODULE(HistProducer);
