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
// Nello Nappi May 9, 2007 Module created
//
//------------------------------------------------------------------------
//
#include "CLHEP/Random/RandomEngine.h"
#include "EvtGenBase/EvtRandomEngine.hh"
#include "GeneratorInterface/EvtGenInterface/interface/myEvtRandomEngine.h"

myEvtRandomEngine::myEvtRandomEngine(CLHEP::HepRandomEngine *xx) {the_engine = xx;}

myEvtRandomEngine::~myEvtRandomEngine() {}

double myEvtRandomEngine::random()
{
  return the_engine->flat();
}
