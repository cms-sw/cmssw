#ifndef Geom_BoundPlane_H
#define Geom_BoundPlane_H

/** \class BoundPlane
 *
 *  A plane in 3D space, with bounds.
 *  
 *  \warning Surfaces are reference counted, so only ReferenceCountingPointer
 *  should be used to point to them. For this reason, they should be 
 *  using the static build() method. 
 *  (The normal constructor will become private in the future).
 */

#include "DataFormats/GeometrySurface/interface/Plane.h"

#endif // Geom_BoundPlane_H
