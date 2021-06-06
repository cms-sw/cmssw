#ifndef SHALLOW_EXAMPLE_PRODUCER
#define SHALLOW_EXAMPLE_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class ShallowExampleProducer : public edm::EDProducer {
public:
  explicit ShallowExampleProducer(const edm::ParameterSet&);
private:
  void produce( edm::Event &, const edm::EventSetup & );
};

#endif
