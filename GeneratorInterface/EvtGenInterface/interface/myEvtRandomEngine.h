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

#include "EvtGenBase/EvtRandomEngine.hh"

namespace CLHEP {
  class HepRandomEngine;
}

class myEvtRandomEngine : public EvtRandomEngine  
{

public:
  
  myEvtRandomEngine(CLHEP::HepRandomEngine* xx);

  ~myEvtRandomEngine() override;

  double random() override;

  void setRandomEngine(CLHEP::HepRandomEngine* v) { the_engine = v; }

  CLHEP::HepRandomEngine* engine() const { return the_engine; }

private:

  void throwNullPtr() const;

  CLHEP::HepRandomEngine* the_engine;
};
#endif
