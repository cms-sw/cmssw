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
}  // namespace edm

class RandomEngineAndDistribution {
public:
  RandomEngineAndDistribution(edm::StreamID const&);
  RandomEngineAndDistribution(edm::LuminosityBlockIndex const&);

  ~RandomEngineAndDistribution();

  CLHEP::HepRandomEngine& theEngine() const { return *engine_; }

  inline double flatShoot(double xmin = 0.0, double xmax = 1.0) const { return xmin + (xmax - xmin) * engine_->flat(); }

  inline double gaussShoot(double mean = 0.0, double sigma = 1.0) const {
    return CLHEP::RandGauss::shoot(engine_, mean, sigma);
  }

  inline unsigned int poissonShoot(double mean) const { return CLHEP::RandPoissonQ::shoot(engine_, mean); }

private:
  CLHEP::HepRandomEngine* engine_;
};
#endif  // FastSimulation_Utilities_RandomEngineAndDistribution_H
