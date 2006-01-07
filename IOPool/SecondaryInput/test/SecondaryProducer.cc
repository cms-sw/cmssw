// File: SecondaryProducer.cc
// Description:  see SecondaryProducer.h
// Author:  Bill Tanenbaum
//
//--------------------------------------------

#include "IOPool/SecondaryInput/test/SecondaryProducer.h"
#include "FWCore/Framework/src/TypeID.h"
#include "FWCore/Framework/interface/BasicHandle.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/src/VectorInputSourceFactory.h"
#include "FWCore/Integration/test/OtherThingCollection.h"
#include "FWCore/Integration/test/ThingCollection.h"

using namespace std;

namespace edm {

  // Constructor 
  SecondaryProducer::SecondaryProducer(const edm::ParameterSet& ps) {
    // make secondary input source
    secInput_ = makeSecInput(ps);

    produces<edmreftest::ThingCollection>();
    produces<edmreftest::OtherThingCollection>("testUserTag");
  }

  // Virtual destructor needed.
  SecondaryProducer::~SecondaryProducer() { }  

  
  // Functions that get called by framework every event
  void SecondaryProducer::produce(edm::Event& e, const edm::EventSetup&) { 

    typedef edmreftest::ThingCollection TC;
    typedef edmreftest::OtherThingCollection OTC;
    typedef edm::Wrapper<TC> WTC;
    typedef edm::Wrapper<OTC> WOTC;

    std::vector<EventPrincipal*> result;
    secInput_->readMany(1, result);
    // secInput_->readMany(e.id().event()-1, 1, result);

    EventPrincipal &p = *result[0];
    EDProduct const* ep = p.getByType(TypeID(typeid(TC))).wrapper();
    assert(ep);
    WTC const* wtp = dynamic_cast<WTC const*>(ep);
    assert(wtp);
    EDProduct const* epo = p.getByType(TypeID(typeid(OTC))).wrapper();
    assert(epo);
    WOTC const* wop = dynamic_cast<WOTC const*>(epo);
    assert(wop);
    TC const* tp = wtp->product();
    OTC const* op = wop->product();
    auto_ptr<TC> thing(new TC(*tp));
    auto_ptr<OTC> otherThing(new OTC(*op));

    // Put output into event
    e.put(thing);
    e.put(otherThing, "testUserTag");
  }


  boost::shared_ptr<VectorInputSource> SecondaryProducer::makeSecInput(ParameterSet const& ps) {
    ParameterSet sec_input = ps.getParameter<ParameterSet>("input");

    boost::shared_ptr<VectorInputSource> input_(static_cast<VectorInputSource *>
      (VectorInputSourceFactory::get()->makeVectorInputSource(sec_input,
      InputSourceDescription()).release()));
    return input_;
  }

} //edm
using edm::SecondaryProducer;
DEFINE_FWK_MODULE(SecondaryProducer)
