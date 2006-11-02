
// $Id$

#include "IOMC/EventVertexGenerators/interface/BaseEventVertexGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/JamesRandom.h"

BaseEventVertexGenerator::BaseEventVertexGenerator(const edm::ParameterSet & p) 
    : m_pBaseEventVertexGenerator(p)
{
   // Get the engine for the current module from the service 
   using namespace edm;
   Service<RandomNumberGenerator> rng;

   if ( ! rng.isAvailable()) {

     throw cms::Exception("Configuration")
       << "The BaseEventVertexGenerator requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }

   CLHEP::HepRandomEngine& engine = rng->getEngine();

   m_engine = &engine;

   // The service owns the engines it provides
   // Only the service should destroy them
   m_ownsEngine = false;
}

BaseEventVertexGenerator::BaseEventVertexGenerator(const edm::ParameterSet & p,
                                                   const long& seed) 
    : m_pBaseEventVertexGenerator(p)
{
   m_engine = new CLHEP::HepJamesRandom(seed);
   m_ownsEngine = true;
}

BaseEventVertexGenerator::~BaseEventVertexGenerator() 
{
   if ( m_ownsEngine ) delete m_engine;
}

CLHEP::HepRandomEngine& BaseEventVertexGenerator::getEngine() 
{
   return *m_engine;
}
