#include "GeneratorInterface/Pythia8Interface/plugins/RandomP8.h"
#include "CLHEP/Random/RandomEngine.h"


CLHEP::HepRandomEngine* randomEngine;


double RandomP8::flat(void)
{
  return randomEngine->flat();
}
