#include <string>

#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Integration/test/OtherThingAlgorithm.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"


namespace edmtest {
  class OtherThingProducer : public edm::EDProducer {
  public:
    explicit OtherThingProducer(edm::ParameterSet const& ps);
    
    virtual ~OtherThingProducer();
    
    virtual void produce(edm::Event& e, edm::EventSetup const& c);
    
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
  private:
    OtherThingAlgorithm alg_;
    edm::EDGetToken thingToken_;
    bool useRefs_;
    bool refsAreTransient_;
  };

  
  OtherThingProducer::OtherThingProducer(edm::ParameterSet const& pset): alg_(), refsAreTransient_(false) {
    produces<OtherThingCollection>("testUserTag");
    useRefs_ = pset.getUntrackedParameter<bool>("useRefs");
    if(useRefs_) {
      thingToken_=consumes<ThingCollection>(pset.getParameter<edm::InputTag>("thingTag"));
    }
    refsAreTransient_ = pset.getUntrackedParameter<bool>("transient");
  }

  // Virtual destructor needed.
  OtherThingProducer::~OtherThingProducer() {}  

  // Functions that gets called by framework every event
  void OtherThingProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::unique_ptr<OtherThingCollection> result(new OtherThingCollection);  //Empty

    // Step C: Get data for algorithm
    edm::Handle<ThingCollection> parentHandle;
    if(useRefs_) {
      bool succeeded = e.getByToken(thingToken_, parentHandle);
      assert(succeeded);
      assert(parentHandle.isValid());
    }

    
    // Step D: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(parentHandle, *result, useRefs_, refsAreTransient_);

    // Step E: Put outputs into event
    e.put(std::move(result), std::string("testUserTag"));
  }
  
  void OtherThingProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("thingTag",edm::InputTag("Thing"))->setComment("Where to get the ThingCollection");
    desc.addUntracked<bool>("useRefs",true)->setComment("Actually get the ThingCollection and build edm::Refs to the contained items.");
    desc.addUntracked<bool>("transient",false)->setComment("If true, then the Refs constructed by the ThingCollection can not be persisted");
    descriptions.add("otherThingProd", desc);
  }

}
using edmtest::OtherThingProducer;
DEFINE_FWK_MODULE(OtherThingProducer);
