#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"

#include "CLHEP/Random/RandomEngine.h"
#include "Randomize.hh"

RandomEngineAndDistribution::RandomEngineAndDistribution(edm::StreamID const& streamID) :
  engine_(nullptr),
  rootEngine_(nullptr) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration") <<
      "RandomNumberGenerator service is not available.\n"
      "You must add the service in the configuration file\n"
      "or remove the module that requires it.";
  }
  engine_ = &rng->getEngine(streamID);

  // define Geant4 engine per thread
  G4Random::setTheEngine(engine_);
}

RandomEngineAndDistribution::RandomEngineAndDistribution(edm::LuminosityBlockIndex const& luminosityBlockIndex) :
  engine_(nullptr),
  rootEngine_(nullptr) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration") <<
      "RandomNumberGenerator service is not available.\n"
      "You must add the service in the configuration file\n"
      "or remove the module that requires it.";
  }
  engine_ = &rng->getEngine(luminosityBlockIndex);

  // define Geant4 engine per thread
  G4Random::setTheEngine(engine_);
}

RandomEngineAndDistribution::~RandomEngineAndDistribution() {
}
