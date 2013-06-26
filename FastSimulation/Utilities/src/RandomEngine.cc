#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"

RandomEngine::RandomEngine(edm::RandomNumberGenerator* rng) 
  : 
  rng_(rng),
  rootEngine_(0)
{
  // The service engine
  engine_ = &(rng->getEngine());
  // Get the TRandom3 egine, to benefit from Root functional random generation
  if ( engine_->name() == "TRandom3" )
    rootEngine_ = ( (edm::TRandomAdaptor*) engine_ )->getRootEngine();
  // If no root engine, use the CLHEP wrapper.
  if ( !rootEngine_ ) { 
    flatDistribution_ = new CLHEP::RandFlat(*engine_);
    gaussianDistribution_ = new CLHEP::RandGaussQ(*engine_);
    poissonDistribution_ = new CLHEP::RandPoissonQ(*engine_);
  }
}

RandomEngine::~RandomEngine() 
{
  if ( !rootEngine_ ) { 
    delete flatDistribution_;
    delete gaussianDistribution_;
    delete poissonDistribution_;  
  }
}




