#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include "TRandom3.h"

RandomEngine::RandomEngine(edm::RandomNumberGenerator* rng) 
  : 
  rng_(rng), 
  rootEngine_(0)
{
  engine_ = &(rng->getEngine()); 
  flatDistribution_ = new CLHEP::RandFlat(*engine_);
  gaussianDistribution_ = new CLHEP::RandGaussQ(*engine_);
  poissonDistribution_ = new CLHEP::RandPoissonQ(*engine_);
}

RandomEngine::RandomEngine(TRandom3* anEngine) 
  : 
  rng_(0), 
  engine_(0),
  rootEngine_(anEngine) 
{}

RandomEngine::~RandomEngine() 
{
  if ( engine_ ) { 
    delete flatDistribution_;
    delete gaussianDistribution_;
    delete poissonDistribution_;  
  }
}

double
RandomEngine::flatShoot(double xmin, double xmax) const{ 
  return engine_? 
    xmin + (xmax-xmin)*flatDistribution_->fire() 
    :
    xmin + (xmax-xmin) * rootEngine_->Rndm();
}

double
RandomEngine::gaussShoot(double mean, double sigma) const { 
  return engine_?
    mean + sigma*gaussianDistribution_->fire()
    : 
    rootEngine_->Gaus(mean,sigma);
}

unsigned int
RandomEngine::poissonShoot(double mean) const{ 
  // return poissonDistribution_->fire(mean);
  // The above line does not work because RandPoissonQ::fire(double) calls
  // the static engine of CLHEP (hence with non-reproducibility effects...)
  // While waiting for a fix, here is the solution:
  if ( engine_ ) { 
    CLHEP::RandPoissonQ aPoissonDistribution(*engine_,mean);
    return aPoissonDistribution.fire();
  } else {  
    return rootEngine_->Poisson(mean);
  }
}



