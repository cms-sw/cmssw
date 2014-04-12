
#include "IOMC/RandomEngine/src/RandomEngineStateProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineStates.h"

#include <memory>

RandomEngineStateProducer::RandomEngineStateProducer(edm::ParameterSet const&) {
  produces<edm::RandomEngineStates, edm::InLumi>("beginLumi");
  produces<edm::RandomEngineStates>();
}

RandomEngineStateProducer::~RandomEngineStateProducer() {
}

void
RandomEngineStateProducer::produce(edm::Event& ev, edm::EventSetup const&) {
  edm::Service<edm::RandomNumberGenerator> randomService;
  if(randomService.isAvailable()) {
    std::auto_ptr<edm::RandomEngineStates> states(new edm::RandomEngineStates);
    states->setRandomEngineStates(randomService->getEventCache());
    ev.put(states);
  }
}

void
RandomEngineStateProducer::beginLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const&) {
  edm::Service<edm::RandomNumberGenerator> randomService;
  if(randomService.isAvailable()) {
    std::auto_ptr<edm::RandomEngineStates> states(new edm::RandomEngineStates);
    states->setRandomEngineStates(randomService->getLumiCache());
    lb.put(states, "beginLumi");
  }
}

void
RandomEngineStateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("randomEngineStateProducer", desc);
}
