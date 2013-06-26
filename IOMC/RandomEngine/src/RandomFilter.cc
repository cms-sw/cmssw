
#include "IOMC/RandomEngine/src/RandomFilter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;

RandomFilter::RandomFilter(edm::ParameterSet const& ps) :
  acceptRate_(ps.getUntrackedParameter<double>("acceptRate")),
  flatDistribution_() {
  Service<RandomNumberGenerator> rng;
  if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "RandomFilter requires the RandomNumberGeneratorService,\n"
         "which is not present in the configuration file.  You must add\n"
         "the service in the configuration file or remove the modules that\n"
         "require it.\n";
  }

  CLHEP::HepRandomEngine& engine = rng->getEngine();

  flatDistribution_.reset(new CLHEP::RandFlat(engine, 0.0, 1.0));
}

RandomFilter::~RandomFilter() {
}

bool RandomFilter::filter(edm::Event&, edm::EventSetup const&) {
  if (flatDistribution_->fire() < acceptRate_) return true;
  return false;
}
