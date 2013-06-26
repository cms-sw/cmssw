#ifndef BoundVolume_H
#define BoundVolume_H

#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

class VolumeBoundary;
class PropagationDirection;

class BoundVolume : public GloballyPositioned<float> {
public:

  BoundVolume( const PositionType& pos, const RotationType& rot) :
    GloballyPositioned<float>( pos, rot) {}

  // virtual vector<const VolumeBoundary*> bounds() const = 0;

  virtual const VolumeBoundary* 
  closestBoundary( const LocalPoint& pos, const LocalVector& momentum,
		   PropagationDirection dir) const = 0;

  virtual const VolumeBoundary* 
  nextBoundary( const LocalPoint& pos, const LocalVector& momentum,
		PropagationDirection dir) const = 0;

};

#endif
