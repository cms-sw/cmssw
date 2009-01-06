#ifndef RNDMEngineAccess_h
#define RNDMEngineAccess_h

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandFlat.h"

//namespace {
  HepRandomEngine& getEngineReference()
  {

   edm::Service<edm::RandomNumberGenerator> rng;
   if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
       << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
          "which appears to be absent.  Please add that service to your configuration\n"
          "or remove the modules that require it.";
   }

// The Service has already instantiated an engine.  Make contact with it.
   return (rng->getEngine());
  }
//}

#endif
