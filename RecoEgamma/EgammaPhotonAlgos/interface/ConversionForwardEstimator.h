#ifndef  RecoEGAMMA_ConversionForwardEstimator_H
#define  RecoEGAMMA_ConversionForwardEstimator_H

/**
 * \class ConversionForwardEstimator
 *  Defines the search area in the  forward 
 *
 *   $Date$ 
 *   $Revision:$
 *  \author Nancy Marinelli, U. of Notre Dame, US
 */

#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h" 
#include "Geometry/Vector/interface/Vector2DBase.h"
#include "Geometry/Vector/interface/LocalTag.h"

#include <iostream> 
class RecHit;
class TrajectoryStateOnSurface;
class BoundPlane;

class ConversionForwardEstimator : public MeasurementEstimator {
public:
  ConversionForwardEstimator() {};
  ConversionForwardEstimator( float phiRangeMin, float phiRangeMax, float dr) :
                           thePhiRangeMin( phiRangeMin), thePhiRangeMax( phiRangeMax), dr_(dr) {
    std::cout << " ConversionForwardEstimator CTOR " << std::endl;
}

  // zero value indicates incompatible ts - hit pair
  virtual std::pair<bool,double> estimate( const TrajectoryStateOnSurface& ts, 
			   const TransientTrackingRecHit& hit) const;
  virtual bool estimate( const TrajectoryStateOnSurface& ts, 
			   const BoundPlane& plane) const;
  virtual ConversionForwardEstimator* clone() const {
    return new ConversionForwardEstimator(*this);
  } 


virtual Local2DVector maximalLocalDisplacement( const TrajectoryStateOnSurface& ts, const BoundPlane& plane) const;

private:

  float thePhiRangeMin;
  float thePhiRangeMax;
  float dr_;

};

#endif // ConversionForwardEstimator_H












