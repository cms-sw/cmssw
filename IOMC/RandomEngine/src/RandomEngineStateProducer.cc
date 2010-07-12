
#include "IOMC/RandomEngine/src/RandomEngineStateProducer.h"

#include <vector>
#include <string>
#include "boost/cstdint.hpp"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

RandomEngineStateProducer::RandomEngineStateProducer(edm::ParameterSet const& iConfig) {
  //register your products
  produces<std::vector<RandomEngineState> >();
}

RandomEngineStateProducer::~RandomEngineStateProducer() {
}

void
RandomEngineStateProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  std::auto_ptr<std::vector<RandomEngineState> > stateVector(new std::vector<RandomEngineState>);

  edm::Service<edm::RandomNumberGenerator> randomService;
  if (randomService.isAvailable()) {
    std::vector<std::string> const& strings = randomService->getCachedLabels();
    std::vector<std::vector<uint32_t> > const& states = randomService->getCachedStates();
    std::vector<std::vector<uint32_t> > const& seeds = randomService->getCachedSeeds();

    std::vector<std::string>::const_iterator iString = strings.begin();
    std::vector<std::vector<uint32_t> >::const_iterator iState = states.begin();
    std::vector<std::vector<uint32_t> >::const_iterator iSeed = seeds.begin();

    for (; iString != strings.end(); ++iString, ++iState, ++iSeed) {
      RandomEngineState engineState;
      engineState.setLabel(*iString);
      engineState.setState(*iState);
      engineState.setSeed(*iSeed);
      stateVector->push_back(engineState);
    }
    iEvent.put(stateVector);
  }
}

void 
RandomEngineStateProducer::beginJob() {
}

void 
RandomEngineStateProducer::endJob() {
}

void
RandomEngineStateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("RandomEngineStateProducer", desc);
}
