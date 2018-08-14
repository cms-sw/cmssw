#ifndef GeneratorInterface_Pythia8Interface_P8RndmEngine_h
#define GeneratorInterface_Pythia8Interface_P8RndmEngine_h

/** \class gen::P8RndmEngine

Description: Used to set an external random number engine
in Pythia 8.  Pythia 8 recognizes a class that has
Pythia8::RndmEngine as a base class and a virtual
flat method.  If you pass a pointer to an object of
this class to Pythia 8 it will use it to generate
random numbers.

Independent of Pythia 8, one can set the CLHEP
engine that this class uses in its flat method.

\author W. David Dagenhart, created 26 November 2013
*/

#include "Pythia8/Basics.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class P8RndmEngine : public Pythia8::RndmEngine {
  public:

    P8RndmEngine() : randomEngine_(nullptr) { }

    // Routine for generating a random number.
    double flat() override;

    void setRandomEngine(CLHEP::HepRandomEngine* v) { randomEngine_ = v; }

  private:

    void throwNullPtr() const;

    CLHEP::HepRandomEngine* randomEngine_;
  };
}
#endif
