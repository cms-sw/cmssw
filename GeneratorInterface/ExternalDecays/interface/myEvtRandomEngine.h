//--------------------------------------------------------------------------
//
// Module: myEvtRandomEngine.hh
//
// Description:
// this is an EvtRandomEngine
// It is used as an interface of the random number engine provided
// by the random number generator service and EvtGen
// Its "random()" method uses the "Flat()" method of the CLHEP::HepRandomEngine
// provided by the Random Number Generator Service
//
// Modification history:
//
//   Nello Nappi     May 9, 2007         Module created
//
//------------------------------------------------------------------------

#ifndef MYEVTRANDOMENGINE_HH
#define MYEVTRANDOMENGINE_HH

//#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Random/RandomEngine.h"
#include "EvtGenBase/EvtRandomEngine.hh"

class myEvtRandomEngine : public EvtRandomEngine  
{

public:
  
  myEvtRandomEngine(CLHEP::HepRandomEngine* xx);

  virtual ~myEvtRandomEngine();

  virtual double random();

private:

  CLHEP::HepRandomEngine* the_engine;

};

#endif

