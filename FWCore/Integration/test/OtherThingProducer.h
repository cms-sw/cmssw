#ifndef Integration_OtherThingProducer_h
#define Integration_OtherThingProducer_h

#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Integration/test/OtherThingAlgorithm.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

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
}

#endif
