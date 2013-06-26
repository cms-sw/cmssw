#ifndef Geom_PlaneBuilder_H
#define Geom_PlaneBuilder_H

#include "DataFormats/GeometrySurface/interface/Plane.h"

class Bounds;

/** \class PlaneBuilder 
 *  This class is deprecated.
 *  Please use the static build() methods in BoundPlane.
 */

class PlaneBuilder {
public:

  typedef Surface::PositionType                  PositionType;
  typedef Surface::RotationType                  RotationType;
  typedef ReferenceCountingPointer<Plane>        ReturnType;

  /** Builds a plane with origin at pos and with rotation matrix rot
   */
  ReturnType plane( const PositionType& pos, const RotationType& rot) const {
    return ReturnType( new Plane( pos, rot));
  }

  /** Same as above, with bounds. The bounds are cloned, and a new 
   *  copy is used in the plane, so you don't have to create them with "new",
   *  and if you do, don't forget to delete them.
   */
  ReturnType plane( const PositionType& pos, const RotationType& rot, 
		    Bounds * bounds) const {
    return ReturnType( new Plane( pos, rot, bounds));
  }

};

#endif
