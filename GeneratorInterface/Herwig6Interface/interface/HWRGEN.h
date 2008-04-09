#ifndef HWRGEN_h
#define HWRGEN_h
#include "CLHEP/Random/RandomEngine.h"

extern CLHEP::HepRandomEngine* randomEngine;

extern "C" {
  double hwrgen_(int*);
}

#endif
