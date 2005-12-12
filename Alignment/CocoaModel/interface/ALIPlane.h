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
#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "OpticalAlignment/CocoaModel/interface/ALILine.h"

class ALIPlane {

public:
  ALIPlane( const Hep3Vector& point, const Hep3Vector& normal );
  // Project a std::vector onto this plane
  Hep3Vector project( const Hep3Vector& vec );
  ALILine lineProject( const Hep3Vector& vec );
  //  Hep3Vector ALIPlane::intersect( const ALIPlane& l2);
  const Hep3Vector& point() const {return _point;};
  const Hep3Vector& normal() const {return _normal;};

private:
  Hep3Vector _point;
  Hep3Vector _normal;

};

#endif
