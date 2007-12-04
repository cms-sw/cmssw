#ifndef FastSimulation_Utilities_RandomEngine_H
#define FastSimulation_Utilities_RandomEngine_H

class TRandom3;

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

  CLHEP::HepRandomEngine* theEngine() const { return engine_; }

  TRandom3* theRootEngine() const { return rootEngine_; }

  RandomEngine(edm::RandomNumberGenerator* rng);

  RandomEngine(TRandom3* anEngine);

  ~RandomEngine();

  double flatShoot(double xmin=0., double xmax=1.) const;
  double gaussShoot(double mean=0., double sigma=1.) const;
  unsigned int poissonShoot(double mean) const;

 private:

  edm::RandomNumberGenerator* rng_;

  CLHEP::RandFlat* flatDistribution_;
  CLHEP::RandGaussQ* gaussianDistribution_;
  CLHEP::RandPoissonQ* poissonDistribution_;
  CLHEP::HepRandomEngine* engine_;

  TRandom3* rootEngine_;

};

#endif // FastSimulation_Utilities_RandomEngine_H
