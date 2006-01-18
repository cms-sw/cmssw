#ifndef Base_PileUp_h
#define Base_PileUp_h

#include <string>
#include <vector>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/TripleRand.h"

namespace edm {
  class PileUp {
  public:
    explicit PileUp(ParameterSet const& pset);
    ~PileUp() {}

    void readPileUp(std::vector<std::vector<Event *> > & result);

    int minBunch() const {return minBunch_;}
    int maxBunch() const {return maxBunch_;}
    double averageNumber() const {return averageNumber_;}
    bool poisson() const {return poisson_;}
    long seed() const {return seed_;}

  private:
    std::string const type_;
    int const minBunch_;
    int const maxBunch_;
    double const averageNumber_;
    int const intAverage_;
    bool const poisson_;
    bool const fixed_;
    bool const none_;
    unsigned int const maxEventsToSkip_;
    long const seed_;
    VectorInputSource * const input_;
    TripleRand eng_;
    RandPoisson poissonDistribution_;
    RandFlat flatDistribution_;
    ModuleDescription md_;
  };
}

#endif
