#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

using namespace gen;

CLHEP::HepRandomEngine& gen::getEngineReference()
{
   edm::Service<edm::RandomNumberGenerator> rng;
   if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
       << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
          "which appears to be absent.  Please add that service to your configuration\n"
          "or remove the modules that require it." << std::endl;
   }

// The Service has already instantiated an engine.  Make contact with it.
   return rng->getEngine();
}
