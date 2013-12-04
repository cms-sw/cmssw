#ifndef FastSimulation_Utilities_RandomEngineAndDistribution_H
#define FastSimulation_Utilities_RandomEngineAndDistribution_H

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "TRandom3.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class LuminosityBlockIndex;
  class StreamID;
}

class RandomEngineAndDistribution {

 public:

  RandomEngineAndDistribution(edm::StreamID const&);
  RandomEngineAndDistribution(edm::LuminosityBlockIndex const&);

  // This is strongly deprecated, it exists for backward compatibility
  // for cases where the above two functions cannot be used easily.
  // It is the intent fix all those cases and delete this function
  // as soon as possible.
  RandomEngineAndDistribution();

  ~RandomEngineAndDistribution();

  CLHEP::HepRandomEngine& theEngine() const { return *engine_; }

  inline double flatShoot(double xmin=0.0, double xmax=1.0) const {
    if(rootEngine_) {
      return xmin + (xmax - xmin) * rootEngine_->Rndm();
    } else {
      CLHEP::RandFlat flatDistribution(*engine_);
      return xmin + (xmax - xmin) * flatDistribution.fire();
    }
  }

  inline double gaussShoot(double mean=0.0, double sigma=1.0) const {
    if(rootEngine_) {
      return rootEngine_->Gaus(mean,sigma);
    } else {
      CLHEP::RandGaussQ gaussianDistribution(*engine_);
      return mean + sigma * gaussianDistribution.fire();
    }
  }

  inline unsigned int poissonShoot(double mean) const{
    if(rootEngine_) {
      return rootEngine_->Poisson(mean);
    } else {
      CLHEP::RandPoissonQ poissonDistribution(*engine_);
      return poissonDistribution.fire(mean);
    }
  }

 private:

  CLHEP::HepRandomEngine* engine_;
  TRandom3* rootEngine_;
};
#endif // FastSimulation_Utilities_RandomEngineAndDistribution_H
