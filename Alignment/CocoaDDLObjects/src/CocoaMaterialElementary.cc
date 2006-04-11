//   COCOA class implementation file
//Id:  CocoaMaterialElementary.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce
#include <map>
#include <fstream>

#include "Alignment/CocoaDDLObjects/interface/CocoaMaterialElementary.h"


CocoaMaterialElementary::CocoaMaterialElementary( ALIstring name, float density, ALIstring symbol, ALIint A, ALIint Z )
{ 

  theName = name;
  theDensity = density;
  theSymbol = symbol;
  theA = A;
  theZ = Z;
}


ALIbool CocoaMaterialElementary::operator==(const CocoaMaterialElementary& mate ) const
{
  float kTolerance = 1.E-9;
  if( mate.getDensity() - theDensity < kTolerance 
      && mate.getSymbol() == theSymbol
      && mate.getA() == theA
      && mate.getZ() == theZ ) {
    return 1;
  } else {
    return 0;
  }

}
