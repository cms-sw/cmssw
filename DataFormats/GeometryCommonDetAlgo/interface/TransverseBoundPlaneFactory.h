#ifndef TransverseBoundPlaneFactory_H
#define TransverseBoundPlaneFactory_H

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

/** Obsolete.
 */

class TransverseBoundPlaneFactory {
public:
  BoundPlane* operator()(const Surface::GlobalPoint& origin, const Surface::GlobalVector& perp) const;
};

#endif
