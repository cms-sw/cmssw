#ifndef TrackerGeometryBuilder_GeomDetLess_H
#define TrackerGeometryBuilder_GeomDetLess_H

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <functional>

/** Defines order of layers in the Tracker as seen by straight tracks
 *  coming from the interaction region.
 */

class GeomDetLess {
public:

  GeomDetLess( PropagationDirection dir = alongMomentum) :
    theDir(dir) {}

  bool operator()( const GeomDet* a, const GeomDet* b) const {
    if (theDir == alongMomentum) return insideOutLess( a, b);
    else return insideOutLess( b, a);
  }

 private:

  PropagationDirection theDir;

  bool insideOutLess( const GeomDet*,const GeomDet*) const;

  bool barrelForwardLess( const GeomDet* blb,
			  const GeomDet* fla) const;

};

#endif
