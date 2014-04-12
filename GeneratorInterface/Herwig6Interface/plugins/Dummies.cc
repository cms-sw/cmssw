#include <CLHEP/Random/RandomEngine.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

CLHEP::HepRandomEngine* randomEngine;

extern "C" {
  void hwaend_() {return;}

  void cmsending_(int* ecode) {
    edm::LogError("")<<"   ERROR: Herwig stoped run after receiving error code "<<*ecode<<".\n";
    throw cms::Exception("Herwig6Error") <<" Herwig stoped run with error code "<<*ecode<<".";
  }

  double hwrgen_(int* idummy)
  {
    return randomEngine->flat();
  }
}
