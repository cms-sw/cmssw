// File: SecondaryProducer.cc
// Description:  see SecondaryProducer.h
// Author:  Bill Tanenbaum
//
//--------------------------------------------

#include "IOPool/SecondaryInput/test/SecondaryProducer.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/Utilities/interface/TypeID.h"

namespace edm {

  // Constructor
  // make secondary input source
  SecondaryProducer::SecondaryProducer(ParameterSet const& pset) :
        secInput_(makeSecInput(pset)),
        sequential_(pset.getUntrackedParameter<bool>("sequential", false)),
        specified_(pset.getUntrackedParameter<bool>("specified", false)),
        firstEvent_(true),
        firstLoop_(true),
        expectedEventNumber_(1) {
    produces<edmtest::ThingCollection>();
    produces<edmtest::OtherThingCollection>("testUserTag");
  }

  // Virtual destructor needed.
  SecondaryProducer::~SecondaryProducer() {}

  // Functions that get called by framework every event
  void SecondaryProducer::produce(Event& e, EventSetup const&) {

    typedef edmtest::ThingCollection TC;
    typedef Wrapper<TC> WTC;

    VectorInputSource::EventPrincipalVector result;
    unsigned int fileSequenceNumber;

    if(sequential_) {
      secInput_->readManySequential(1, result, fileSequenceNumber);
      if(result.empty()) {
        secInput_->rewind();
        secInput_->readManySequential(1, result, fileSequenceNumber);
      }
    } else if(specified_) {
      std::vector<EventID> events;
      // Just for simplicity, we use the event ID from the primary to read the secondary.
      events.push_back(e.id());
      secInput_->readManySpecified(events, result);
    } else {
      secInput_->readManyRandom(1, result, fileSequenceNumber);
    }

    EventPrincipal *p = &**result.begin();
    EventNumber_t en = p->id().event();
    BasicHandle bh = p->getByType(TypeID(typeid(TC)));
    assert(bh.isValid());
    if(!(bh.interface()->dynamicTypeInfo() == typeid(TC))) {
      handleimpl::throwConvertTypeError(typeid(TC), bh.interface()->dynamicTypeInfo());
    }
    WTC const* wtp = static_cast<WTC const*>(bh.wrapper());

    assert(wtp);
    TC const* tp = wtp->product();
    std::auto_ptr<TC> thing(new TC(*tp));

    // Put output into event
    e.put(thing);

    if(!sequential_ && !specified_ && firstLoop_ && en == 1) {
      expectedEventNumber_ = 1;
      firstLoop_ = false;
    }
    if(firstEvent_) {
      firstEvent_ = false;
      if(!sequential_ && !specified_) {
        expectedEventNumber_ = en;
      }
    }
    assert (expectedEventNumber_ == en);
    ++expectedEventNumber_;
  }

  boost::shared_ptr<VectorInputSource> SecondaryProducer::makeSecInput(ParameterSet const& ps) {
    ParameterSet const& sec_input = ps.getParameterSet("input");

    boost::shared_ptr<VectorInputSource> input_(static_cast<VectorInputSource *>
      (VectorInputSourceFactory::get()->makeVectorInputSource(sec_input,
      InputSourceDescription()).release()));
    return input_;
  }

} //edm
using edm::SecondaryProducer;
DEFINE_FWK_MODULE(SecondaryProducer);
