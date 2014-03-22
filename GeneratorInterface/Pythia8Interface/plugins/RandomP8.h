#include "Pythia8/Pythia.h"
#include "CLHEP/Random/RandomEngine.h"

extern CLHEP::HepRandomEngine* randomEngine;

class RandomP8 : public Pythia8::RndmEngine {

  public:

    // Constructor.
    RandomP8() {;}

    // Routine for generating a random number.
    double flat();

  private:

};
