#ifndef Geom_BoundCylinder_H
#define Geom_BoundCylinder_H

/** \class BoundCylinder
 *
 *  A Cylinder with Bounds.
 *
 *  \warning Surfaces are reference counted, so only ReferenceCountingPointer
 *  should be used to point to them. For this reason, they should be 
 *  using the static build() method. 
 *  (The normal constructor will become private in the future).
 *
 *  $Date: 2012/12/23 16:33:37 $
 *  $Revision: 1.5 $
 */

#include "DataFormats/GeometrySurface/interface/Cylinder.h"

using BoundCylinder = Cylinder;


#endif // Geom_BoundCylinder_H
