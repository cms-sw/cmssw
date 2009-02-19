#include "CLHEP/Random/RandomEngine.h"

CLHEP::HepRandomEngine* _BeamHalo_randomEngine;

extern "C" {
  double bhgpyr_(int* idummy)
  {
    return _BeamHalo_randomEngine->flat();
  }
}

