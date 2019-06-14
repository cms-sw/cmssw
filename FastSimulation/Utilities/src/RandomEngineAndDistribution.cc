#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"

#include "CLHEP/Random/RandomEngine.h"

RandomEngineAndDistribution::RandomEngineAndDistribution(edm::StreamID const& streamID) : engine_(nullptr) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "RandomNumberGenerator service is not available.\n"
                                             "You must add the service in the configuration file\n"
                                             "or remove the module that requires it.";
  }
  engine_ = &rng->getEngine(streamID);
}

RandomEngineAndDistribution::RandomEngineAndDistribution(edm::LuminosityBlockIndex const& luminosityBlockIndex)
    : engine_(nullptr) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "RandomNumberGenerator service is not available.\n"
                                             "You must add the service in the configuration file\n"
                                             "or remove the module that requires it.";
  }
  engine_ = &rng->getEngine(luminosityBlockIndex);
}

RandomEngineAndDistribution::~RandomEngineAndDistribution() {}
