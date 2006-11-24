#ifndef FastSimulation_Utilities_RandomEngine_H
#define FastSimulation_Utilities_RandomEngine_H

namespace CLHEP { 
  class RandFlat;
  class RandGaussQ;
  class RandPoissonQ;
}

namespace edm {
  class RandomNumberGenerator;
}

class RandomEngine {

public:

  edm::RandomNumberGenerator* theRandomNumberGenerator() const {return rng_;}

  static RandomEngine* instance(edm::RandomNumberGenerator* rng) ;
  static RandomEngine* instance() ;

  ~RandomEngine();

  double flatShoot(double xmin=0., double xmax=1.);
  double gaussShoot(double mean=0., double sigma=1.);
  double poissonShoot(double mean);

private:

  RandomEngine(edm::RandomNumberGenerator* rng);
  static RandomEngine* myself;
  edm::RandomNumberGenerator* rng_;

  CLHEP::RandFlat* flatDistribution_;
  CLHEP::RandGaussQ* gaussianDistribution_;
  CLHEP::RandPoissonQ* poissonDistribution_;

};

#endif // FastSimulation_Utilities_RandomEngine_H
