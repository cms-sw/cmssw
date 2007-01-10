#ifndef Integration_OtherThingProducer_h
#define Integration_OtherThingProducer_h

#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Integration/test/OtherThingAlgorithm.h"

namespace edmtest {
  class OtherThingProducer : public edm::EDProducer {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit OtherThingProducer(edm::ParameterSet const& ps);

    virtual ~OtherThingProducer();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

    virtual void beginRun(edm::Run& r, edm::EventSetup const& c);

    virtual void endRun(edm::Run& r, edm::EventSetup const& c);

    virtual void beginLuminosityBlock(edm::LuminosityBlock& lb, edm::EventSetup const& c);

    virtual void endLuminosityBlock(edm::LuminosityBlock& lb, edm::EventSetup const& c);

  private:
    OtherThingAlgorithm alg_;
    std::string thingLabel_;
  };
}

#endif
