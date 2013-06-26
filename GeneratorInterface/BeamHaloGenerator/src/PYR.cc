#include "CLHEP/Random/RandomEngine.h"

CLHEP::HepRandomEngine* _BeamHalo_randomEngine;

extern "C" {
  float bhgpyr_(int* idummy)
  {
    return (float)_BeamHalo_randomEngine->flat();
  }
}

