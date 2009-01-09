#ifndef Integration_ProducerWithPSetDesc_h
#define Integration_ProducerWithPSetDesc_h

// Used to test the ParameterSetDescription.
// This module has a description with many
// different types and values of parameters,
// including nested ParameterSets and vectors
// of them.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {
  class ConfigurationDescriptions;
}

namespace edmtest {
  class ProducerWithPSetDesc : public edm::EDProducer {
  public:

    explicit ProducerWithPSetDesc(edm::ParameterSet const& ps);

    virtual ~ProducerWithPSetDesc();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  };
}
#endif
