#ifndef BoundarySurface_H
#define BoundarySurface_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include <vector>

class VolumeBoundary;

class BoundarySurface {
public:

  enum Side {positive, negative};

  virtual std::vector<const VolumeBoundary*> volumeBoundaries( Side) const = 0;

  virtual const VolumeBoundary* volumeBoundary( const LocalPoint&, Side) const = 0;

};

#endif
