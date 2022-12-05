
#include "RandomEngineStateProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineStates.h"

#include <memory>

RandomEngineStateProducer::RandomEngineStateProducer(edm::ParameterSet const&) {
  produces<edm::RandomEngineStates, edm::Transition::BeginLuminosityBlock>("beginLumi");
  produces<edm::RandomEngineStates>();
}

RandomEngineStateProducer::~RandomEngineStateProducer() {}

void RandomEngineStateProducer::produce(edm::StreamID iID, edm::Event& ev, edm::EventSetup const&) const {
  edm::Service<edm::RandomNumberGenerator> randomService;
  if (randomService.isAvailable()) {
    auto states = std::make_unique<edm::RandomEngineStates>();
    states->setRandomEngineStates(randomService->getEventCache(ev.streamID()));
    ev.put(std::move(states));
  }
}

void RandomEngineStateProducer::globalBeginLuminosityBlockProduce(edm::LuminosityBlock& lb,
                                                                  edm::EventSetup const&) const {
  edm::Service<edm::RandomNumberGenerator> randomService;
  if (randomService.isAvailable()) {
    auto states = std::make_unique<edm::RandomEngineStates>();
    states->setRandomEngineStates(randomService->getLumiCache(lb.index()));
    lb.put(std::move(states), "beginLumi");
  }
}

void RandomEngineStateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("randomEngineStateProducer", desc);
}
