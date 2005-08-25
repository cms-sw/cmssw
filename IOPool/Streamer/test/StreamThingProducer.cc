
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Streamer/interface/StreamTestThing.h"
#include "IOPool/Streamer/test/StreamThingProducer.h"

using namespace std;
using namespace edmtestprod;

namespace edmtest_thing
{
  StreamThingProducer::StreamThingProducer(edm::ParameterSet const& ps):
    size_(ps.getParameter<int>("array_size"))
  {
    produces<edmtestprod::StreamTestThing>();
  }

  StreamThingProducer::~StreamThingProducer()
  {
  }  

  // Functions that gets called by framework every event
  void StreamThingProducer::produce(edm::Event& e, edm::EventSetup const&)
  {
    std::auto_ptr<StreamTestThing> result(new StreamTestThing(size_));
    e.put(result);
  }
}

using edmtest_thing::StreamThingProducer;
DEFINE_FWK_MODULE(StreamThingProducer)
