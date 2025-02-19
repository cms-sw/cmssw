//  COCOA class header file
//Id:  ALIPlane.h
//CAT: Fit
//
//   Base class for planes, defined by a point and the plane normal direction
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _ALIPlane_HH
#define _ALIPlane_HH
#include <CLHEP/Vector/ThreeVector.h>
#include "Alignment/CocoaModel/interface/ALILine.h"

class ALIPlane {

public:
  ALIPlane( const CLHEP::Hep3Vector& point, const CLHEP::Hep3Vector& normal );
  // Project a std::vector onto this plane
  CLHEP::Hep3Vector project( const CLHEP::Hep3Vector& vec );
  ALILine lineProject( const CLHEP::Hep3Vector& vec );
  //  CLHEP::Hep3Vector ALIPlane::intersect( const ALIPlane& l2);
  const CLHEP::Hep3Vector& point() const {return _point;};
  const CLHEP::Hep3Vector& normal() const {return _normal;};

private:
  CLHEP::Hep3Vector _point;
  CLHEP::Hep3Vector _normal;

};

#endif
