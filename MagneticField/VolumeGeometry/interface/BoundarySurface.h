#ifndef BoundarySurface_H
#define BoundarySurface_H

#include <vector>

class VolumeBoundary;

class BoundarySurface {
public:

  enum Side {positive, negative};

  vector<const VolumeBoundary*> volumeBoundaries( Side) const = 0;

  const VolumeBoundary* volumeBoundary( const LocalPoint&, Side) const = 0;

};

#endif
