#ifndef BoundingBox_H
#define BoundingBox_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

class BoundPlane;

/** A helper class that returns the corners of a rectangle that
 *  fully contains a bound plane.
 */

class BoundingBox {
public:

  std::vector<GlobalPoint> corners( const BoundPlane&);

};

#endif
