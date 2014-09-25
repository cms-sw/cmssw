
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#if 1
#include "DataFormats/TestObjects/interface/StreamTestThing.h"
#include "DataFormats/TestObjects/interface/StreamTestTmpl.h"
typedef edmtestprod::StreamTestThing WriteThis;
#else
#include "FWCore/Integration/interface/IntArray.h"
typedef edmtestprod::IntArray WriteThis;
#endif

#include "IOPool/Streamer/test/StreamThingProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 

#include <sstream>

using namespace edmtestprod;


namespace edmtest_thing
{
  typedef StreamTestTmpl<OSimple> TestDbl;

  StreamThingProducer::StreamThingProducer(edm::ParameterSet const& ps):
    size_(ps.getParameter<int>("array_size")),
    inst_count_(ps.getParameter<int>("instance_count")),
    start_count_(ps.getUntrackedParameter<int>("start_count",0)),
    apply_bit_mask_(ps.getUntrackedParameter<bool>("apply_bit_mask",false)),
    bit_mask_(ps.getUntrackedParameter<uint32_t>("bit_mask",0))
  {
    for(int i=0;i<inst_count_;++i) {
	std::ostringstream ost;
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
      std::unique_ptr<WriteThis> result(new WriteThis(size_));

        // The purpose of this masking is to allow
        // some limited control of how much smaller these
        // vectors will get when compressed.  The more bits
        // are set to zero the more effect compression will have.
        if (apply_bit_mask_) {
          for (int j = 0; j < size_; ++j) {
            result->data_.at(j) &= bit_mask_;
          }
        }

      e.put(std::move(result),names_[i]);
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
