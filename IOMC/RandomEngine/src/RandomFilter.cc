
#include "IOMC/RandomEngine/src/RandomFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandomEngine.h"

using namespace edm;

RandomFilter::RandomFilter(edm::ParameterSet const& ps) :
  acceptRate_(ps.getUntrackedParameter<double>("acceptRate")) {
  Service<RandomNumberGenerator> rng;
  if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "RandomFilter requires the RandomNumberGeneratorService,\n"
         "which is not present in the configuration file.  You must add\n"
         "the service in the configuration file or remove the modules that\n"
         "require it.\n";
  }
}

RandomFilter::~RandomFilter() {
}

bool RandomFilter::filter(edm::Event& event, edm::EventSetup const&) {
  Service<RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine(event.streamID());
  if (engine.flat() < acceptRate_) return true;
  return false;
}
