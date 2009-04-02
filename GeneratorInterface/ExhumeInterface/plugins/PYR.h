#ifndef PYR_h
#define PYR_h
#include "CLHEP/Random/RandomEngine.h"

extern CLHEP::HepRandomEngine* randomEngine;

extern "C" {
  double pyr_(int*);
}

#endif
