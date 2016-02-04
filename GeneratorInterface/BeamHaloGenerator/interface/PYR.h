#ifndef PYR_h
#define PYR_h
#include "CLHEP/Random/RandomEngine.h"

extern CLHEP::HepRandomEngine* _BeamHalo_randomEngine;

extern "C" {
  double bhgpyr_(int*);
}

#endif
