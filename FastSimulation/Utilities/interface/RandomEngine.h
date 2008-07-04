#ifndef FastSimulation_Utilities_RandomEngine_H
#define FastSimulation_Utilities_RandomEngine_H

#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "TRandom3.h"

namespace CLHEP { 
  class HepRandomEngine;
}

namespace edm {
  class RandomNumberGenerator;
}

class RandomEngine {

 public:

  edm::RandomNumberGenerator* theRandomNumberGenerator() const {return rng_;}

  CLHEP::HepRandomEngine* theEngine() const { return engine_; }

  RandomEngine(edm::RandomNumberGenerator* rng);

  ~RandomEngine();

  inline double flatShoot(double xmin=0.0, double xmax=1.0) const{ 
    return rootEngine_ ? 
      xmin + (xmax-xmin) * rootEngine_->Rndm()
      :
      xmin + (xmax-xmin)*flatDistribution_->fire();
  }

  inline double gaussShoot(double mean=0.0, double sigma=1.0) const { 
    return rootEngine_ ?
      rootEngine_->Gaus(mean,sigma)
      : 
      mean + sigma*gaussianDistribution_->fire();
  }
  
  inline unsigned int poissonShoot(double mean) const{ 
    return rootEngine_ ? 
      rootEngine_->Poisson(mean)
      :
      poissonDistribution_->fire(mean);
  }
  
 private:

  edm::RandomNumberGenerator* rng_;

  CLHEP::RandFlat* flatDistribution_;
  CLHEP::RandGauss* gaussianDistribution_;
  CLHEP::RandPoisson* poissonDistribution_;
  CLHEP::HepRandomEngine* engine_;

  TRandom3* rootEngine_;

};

#endif // FastSimulation_Utilities_RandomEngine_H
