#ifndef FastSimulation_Utilities_RandomEngine_H
#define FastSimulation_Utilities_RandomEngine_H

#include <map>

namespace CLHEP { 
  class RandFlat;
  class RandGaussQ;
  class RandPoissonQ;
  class HepRandomEngine;
}

namespace edm {
  class RandomNumberGenerator;
}

class RandomEngine {

public:

  edm::RandomNumberGenerator* theRandomNumberGenerator() const {return rng_;}

  CLHEP::HepRandomEngine* theEngine() const { return engine; }

  RandomEngine(edm::RandomNumberGenerator* rng);

  ~RandomEngine();

  double flatShoot(double xmin=0., double xmax=1.) const;
  double gaussShoot(double mean=0., double sigma=1.) const;
  unsigned int poissonShoot(double mean) const;

private:

  edm::RandomNumberGenerator* rng_;

  CLHEP::RandFlat* flatDistribution_;
  CLHEP::RandGaussQ* gaussianDistribution_;
  CLHEP::RandPoissonQ* poissonDistribution_;
  CLHEP::HepRandomEngine* engine;

};

#endif // FastSimulation_Utilities_RandomEngine_H
