
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#if 1
#include "DataFormats/TestObjects/interface/StreamTestThing.h"
#include "DataFormats/TestObjects/interface/StreamTestTmpl.h"
#include "DataFormats/TestObjects/interface/StreamTestSimple.h"
typedef edmtestprod::StreamTestThing WriteThis;
#else
#include "FWCore/Integration/interface/IntArray.h"
typedef edmtestprod::IntArray WriteThis;
#endif

#include "IOPool/Streamer/test/StreamThingProducer.h"

#include <sstream>

using namespace std;
using namespace edmtestprod;


namespace edmtest_thing
{
  typedef StreamTestTmpl<OSimple> TestDbl;

  StreamThingProducer::StreamThingProducer(edm::ParameterSet const& ps):
    size_(ps.getParameter<int>("array_size")),
    inst_count_(ps.getParameter<int>("instance_count")),
    start_count_(ps.getUntrackedParameter<int>("start_count",0))
  {
    for(int i=0;i<inst_count_;++i) {
	ostringstream ost;
	ost << (i+start_count_);
	names_.push_back(ost.str());
	produces<WriteThis>(ost.str());
    }

    // produces<TestDbl>();
    //produces<StreamTestSimple>();
    // produces<Pig>();
  }

  StreamThingProducer::~StreamThingProducer()
  {
  }  

  // Functions that gets called by framework every event
  void StreamThingProducer::produce(edm::Event& e, edm::EventSetup const&)
  {
    for(int i = 0; i < inst_count_; ++i) {
	std::auto_ptr<WriteThis> result(new WriteThis(size_));
	e.put(result,names_[i]);
    }

    //std::auto_ptr<TestDbl> d(new TestDbl);
    //e.put(d);
    //std::auto_ptr<StreamTestSimple> d1(new StreamTestSimple);
    //e.put(d1);
    //std::auto_ptr<Pig> d1(new Pig);
    //e.put(d1);
  }
}

using edmtest_thing::StreamThingProducer;
DEFINE_FWK_MODULE(StreamThingProducer);
