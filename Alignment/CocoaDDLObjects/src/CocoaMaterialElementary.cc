//   COCOA class implementation file
//Id:  CocoaMaterialElementary.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce
#include <map>
#include <fstream>
#include <cmath>		// include floating-point std::abs functions

#include "Alignment/CocoaDDLObjects/interface/CocoaMaterialElementary.h"


CocoaMaterialElementary::CocoaMaterialElementary( ALIstring name, float density, ALIstring symbol, float A, ALIint Z ) :
  theName(name),
  theDensity(density),
  theSymbol(symbol),
  theA(A),
  theZ(Z)
{
}


ALIbool CocoaMaterialElementary::operator==(const CocoaMaterialElementary& mate ) const
{
  // GM: Using numeric_limits<float>::epsilon() might be better instead of a
  //     magic number 'kTolerance'. Not changing this to not break code
  //     potentially relying on this number.
  const float kTolerance = 1.E-9;
  return ( std::abs(mate.getDensity() - theDensity) < kTolerance
	   && mate.getSymbol() == theSymbol
	   && std::abs(mate.getA() - theA) < kTolerance
	   && mate.getZ() == theZ );
}
