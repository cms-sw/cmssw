//   COCOA class implementation file
//Id:  ALIPlane.cc
//CAT: Fit
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/ALIPlane.h"
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h" 
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Constructor
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIPlane::ALIPlane( const CLHEP::Hep3Vector& point, const CLHEP::Hep3Vector& normal ) 
: _point(point)
{
  _normal = normal * (1. / normal.mag());
  if (ALIUtils::debug >= 5) {
    ALIUtils::dump3v( _point, " Created ALIplane: point");
    ALIUtils::dump3v( _normal, " Created ALIPlane: normal");
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Project a std::vector onto this plane:
//@@   Project on normal to plane and substract this projection to vec
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CLHEP::Hep3Vector ALIPlane::project( const CLHEP::Hep3Vector& vec ) 
{
  //---------- Project vec on normal to plane 
  ALIdouble proj = vec.dot(_normal) * (1. / vec.mag() );
  //---------- Substract this projection to vec
  CLHEP::Hep3Vector vecproj = vec - (proj * _normal);
  //-  ALIUtils::dump3v( _normal, "plane _normal");
  //- std::cout << " proj on normal " << proj << std::endl;
  //- ALIUtils::dump3v( vec , "std::vector");
  return vecproj;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Project a std::vector onto this plane:
//@@   Project on normal to plane and substract this projection to vec
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALILine ALIPlane::lineProject( const CLHEP::Hep3Vector& vec ) 
{
  //---------- Project vec on normal to plane 
  ALIdouble proj = vec.dot(_normal) * (1. / vec.mag() );
  //---------- Substract this projection to vec
  CLHEP::Hep3Vector vecproj = vec - (proj * _normal);
  //-  ALIUtils::dump3v( _normal, "plane _normal");
  //- std::cout << " proj on normal " << proj << std::endl;
  //- ALIUtils::dump3v( vec , "std::vector");
  return ALILine( this->_point, vecproj );
}

