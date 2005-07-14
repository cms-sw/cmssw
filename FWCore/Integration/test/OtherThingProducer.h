#ifndef EDMREFTEST_OTHERTHINGPRODUCER_H
#define EDMREFTEST_OTHERTHINGPRODUCER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Integration/test/OtherThingAlgorithm.h"

namespace edm {
  class ParameterSet;
}

namespace edmreftest {
  class OtherThingProducer : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit OtherThingProducer(edm::ParameterSet const& ps);

    virtual ~OtherThingProducer();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    OtherThingAlgorithm alg_;
  };
}

#endif
