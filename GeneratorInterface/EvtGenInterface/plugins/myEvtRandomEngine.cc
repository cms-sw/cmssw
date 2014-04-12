//--------------------------------------------------------------------------
//
// Module: myEvtRandomEngine.cpp
//
// Description:
// this is an EvtRandomEngine
// It is used as an interface of the random number engine provided
// by the CMSSW Random Number Generator Service and EvtGen
// Its "random()" method uses the "Flat()" method of the CLHEP::HepRandomEngine
// provided by the Random Number Generator Service
//
// Modification history:
//
//   Nello Nappi     May 9, 2007         Module created
//
//------------------------------------------------------------------------
//
#include "GeneratorInterface/EvtGenInterface/interface/myEvtRandomEngine.h"
#include "CLHEP/Random/RandomEngine.h"
#include "FWCore/Utilities/interface/EDMException.h"

myEvtRandomEngine::myEvtRandomEngine(CLHEP::HepRandomEngine *xx) {the_engine = xx;}

myEvtRandomEngine::~myEvtRandomEngine() {}

double myEvtRandomEngine::random()
{
  if(the_engine == nullptr) {
    throwNullPtr();
  }
  return the_engine->flat();
}

void myEvtRandomEngine::throwNullPtr() const {
  throw edm::Exception(edm::errors::LogicError)
    << "The EvtGen code attempted to a generate random number while\n"
    << "the engine pointer was null. This might mean that the code\n"
    << "was modified to generate a random number outside the event and\n"
    << "beginLuminosityBlock methods, which is not allowed.\n";
}
