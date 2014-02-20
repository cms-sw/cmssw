#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"

#include "CLHEP/Random/RandomEngine.h"

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

  // Get the TRandom3 egine, to benefit from Root functional random generation
  if ( engine_->name() == "TRandom3" )
    rootEngine_ = ( (edm::TRandomAdaptor*) engine_ )->getRootEngine();
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

  // Get the TRandom3 egine, to benefit from Root functional random generation
  if ( engine_->name() == "TRandom3" )
    rootEngine_ = ( (edm::TRandomAdaptor*) engine_ )->getRootEngine();
}

RandomEngineAndDistribution::RandomEngineAndDistribution() :
  engine_(nullptr),
  rootEngine_(nullptr) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration") <<
      "RandomNumberGenerator service is not available.\n"
      "You must add the service in the configuration file\n"
      "or remove the module that requires it.";
  }
  engine_ = &rng->getEngine();

  // Get the TRandom3 egine, to benefit from Root functional random generation
  if ( engine_->name() == "TRandom3" )
    rootEngine_ = ( (edm::TRandomAdaptor*) engine_ )->getRootEngine();
}

RandomEngineAndDistribution::~RandomEngineAndDistribution() {
}
