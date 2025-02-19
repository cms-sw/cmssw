#include "HWRGEN.h"
#include "CLHEP/Random/RandomEngine.h"

CLHEP::HepRandomEngine* randomEngine;

extern "C" {
  double hwrgen_(int* idummy)
  {
    return randomEngine->flat();
  }
}

