#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

RandomEngine::RandomEngine(edm::RandomNumberGenerator* rng) : rng_(rng) 
{
  CLHEP::HepRandomEngine& engine = rng->getEngine(); 
  flatDistribution_ = new CLHEP::RandFlat(engine);
  gaussianDistribution_ = new CLHEP::RandGaussQ(engine);
  poissonDistribution_ = new CLHEP::RandPoissonQ(engine);
}

RandomEngine::~RandomEngine() 
{
  delete flatDistribution_;
  delete gaussianDistribution_;
  delete poissonDistribution_;
}

double
RandomEngine::flatShoot(double xmin, double xmax) const{ 
  return xmin + (xmax-xmin)*flatDistribution_->fire();
}

double
RandomEngine::gaussShoot(double mean, double sigma) const { 
  return mean + sigma*gaussianDistribution_->fire();
}

double
RandomEngine::poissonShoot(double mean) const { 
  return poissonDistribution_->fire(mean);
}



