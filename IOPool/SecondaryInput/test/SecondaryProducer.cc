// File: SecondaryProducer.cc
// Description:  see SecondaryProducer.h
// Author:  Bill Tanenbaum
//
//--------------------------------------------

#include "IOPool/SecondaryInput/test/SecondaryProducer.h"
#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"

namespace edm {

  // Constructor 
  // make secondary input source
  SecondaryProducer::SecondaryProducer(const edm::ParameterSet& ps) : secInput_(makeSecInput(ps)) {
    produces<edmtest::ThingCollection>();
    produces<edmtest::OtherThingCollection>("testUserTag");
  }

  // Virtual destructor needed.
  SecondaryProducer::~SecondaryProducer() { }  

  
  // Functions that get called by framework every event
  void SecondaryProducer::produce(edm::Event& e, const edm::EventSetup&) { 

    typedef edmtest::ThingCollection TC;
    typedef edm::Wrapper<TC> WTC;

    VectorInputSource::EventPrincipalVector result;
    unsigned int fileSequenceNumber;
    secInput_->readManyRandom(1, result, fileSequenceNumber);

    EventPrincipal *p = &**result.begin();
    EDProduct const* ep = p->getByType(TypeID(typeid(TC))).wrapper();
    assert(ep);
    WTC const* wtp = dynamic_cast<WTC const*>(ep);
    assert(wtp);
    TC const* tp = wtp->product();
    std::auto_ptr<TC> thing(new TC(*tp));

    // Put output into event
    e.put(thing);
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
