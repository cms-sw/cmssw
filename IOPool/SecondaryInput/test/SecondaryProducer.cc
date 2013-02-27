// File: SecondaryProducer.cc
// Description:  see SecondaryProducer.h
// Author:  Bill Tanenbaum
//
//--------------------------------------------

#include "IOPool/SecondaryInput/test/SecondaryProducer.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "boost/bind.hpp"

#include <memory>
#include <string>

namespace edm {

  // Constructor
  // make secondary input source
  SecondaryProducer::SecondaryProducer(ParameterSet const& pset) :
        secInput_(makeSecInput(pset)),
        sequential_(pset.getUntrackedParameter<bool>("sequential", false)),
        specified_(pset.getUntrackedParameter<bool>("specified", false)),
        lumiSpecified_(pset.getUntrackedParameter<bool>("lumiSpecified", false)),
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
    if(sequential_) {
      if(lumiSpecified_) {
        // Just for simplicity, we use the luminosity block ID from the primary to read the secondary.
        secInput_->loopSequentialWithID(LuminosityBlockID(e.id().run(), e.id().luminosityBlock()), 1, boost::bind(&SecondaryProducer::processOneEvent, this, _1, boost::ref(e)));
      } else {
        secInput_->loopSequential(1, boost::bind(&SecondaryProducer::processOneEvent, this, _1, boost::ref(e)));
      }
    } else if(specified_) {
      // Just for simplicity, we use the event ID from the primary to read the secondary.
      std::vector<EventID> events(1, e.id());
      secInput_->loopSpecified(events, boost::bind(&SecondaryProducer::processOneEvent, this, _1, boost::ref(e)));
    } else {
      if(lumiSpecified_) {
        // Just for simplicity, we use the luminosity block ID from the primary to read the secondary.
        secInput_->loopRandomWithID(LuminosityBlockID(e.id().run(), e.id().luminosityBlock()), 1, boost::bind(&SecondaryProducer::processOneEvent, this, _1, boost::ref(e)));
      } else {
        secInput_->loopRandom(1, boost::bind(&SecondaryProducer::processOneEvent, this, _1, boost::ref(e)));
      }
    }
  }

  void SecondaryProducer::processOneEvent(EventPrincipal const& eventPrincipal, Event& e) {
    typedef edmtest::ThingCollection TC;
    typedef Wrapper<TC> WTC;

    EventNumber_t en = eventPrincipal.id().event();
    // Check that secondary source products are retrieved from the same event as the EventAuxiliary
    BasicHandle bhandle = eventPrincipal.getByLabel(PRODUCT_TYPE, TypeID(typeid(edmtest::IntProduct)),
                                                    "EventNumber",
                                                    "",
                                                    "");
    assert(bhandle.isValid());
    Handle<edmtest::IntProduct> handle;
    convert_handle<edmtest::IntProduct>(bhandle, handle);
    assert(static_cast<EventNumber_t>(handle->value) == en);

    // Check that primary source products are retrieved from the same event as the EventAuxiliary
    e.getByLabel<edmtest::IntProduct>("EventNumber", handle);
    assert(static_cast<EventNumber_t>(handle->value) == e.id().event());

    BasicHandle bh = eventPrincipal.getByLabel(PRODUCT_TYPE, TypeID(typeid(TC)),
                                               "Thing",
                                               "",
                                               "");
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
