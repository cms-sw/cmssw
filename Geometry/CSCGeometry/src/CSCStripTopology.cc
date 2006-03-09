#include "Geometry/CSCGeometry/interface/CSCStripTopology.h"

#include <iostream>

int CSCStripTopology::nearestStrip(const LocalPoint & lp) const
{
  // xxxStripTopology::strip() is expected to have range 0. to 
  // float(no of strips), but be extra careful and enforce that

  float fstrip = this->strip(lp);
  int n_strips = this->nstrips();
  fstrip = ( fstrip>=0. ? fstrip :  0. ); // enforce minimum 0.
  int near = static_cast<int>( fstrip ) + 1; // first strip is 1
// enforce maximum at right edge
  near = ( near<=n_strips ? near : n_strips );
  return near;
}

std::ostream & operator<<( std::ostream & os, const CSCStripTopology & r )
{
  return r.put(os);
}
