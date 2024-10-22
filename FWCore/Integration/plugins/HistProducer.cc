#include "TH1F.h"

#include "HistProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edmtest {
  HistProducer::HistProducer(edm::ParameterSet const&) {
    produces<TH1F>();
    //produces<ThingWithHist>();
  }

  // Functions that gets called by framework every event
  void HistProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    //Empty Histograms
    e.put(std::make_unique<TH1F>());
    //e.put(std::make_unique<ThingWithHist>());
  }

}  // namespace edmtest
using edmtest::HistProducer;
DEFINE_FWK_MODULE(HistProducer);
