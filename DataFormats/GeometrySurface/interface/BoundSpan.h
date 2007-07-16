#ifndef Geom_BoundSpan_H
#define Geom_BoundSpan_H

/**
 *  compute the span of a bound surface in the global space
 *
 *
 */

#include <utility>
class BoundSurface;

namespace boundSpan {
  
  pair<float, float> 
  computePhiSpan( const BoundSurface& plane);

}

#endif
