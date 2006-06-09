#ifndef RecoEGAMMA_ConversionBarrelEstimator_H
#define RecoEGAMMA_ConversionBarrelEstimator_H
/**
 * \class ConversionBarrelEstimator
 *  Defines the search area in the barrel 
 *
 *   $Date:  $
 *   $Revision:  $
 *   \author Nancy Marinelli, U. of Notre Dame, US
 */

#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h" 
#include "Geometry/Vector/interface/Vector2DBase.h"
#include "Geometry/Vector/interface/LocalTag.h"


class TrajectoryStateOnSurface;
class RecHit;
class BoundPlane;

class ConversionBarrelEstimator : public MeasurementEstimator {
public:
  ConversionBarrelEstimator() {};
  ConversionBarrelEstimator( float phiRangeMin, float phiRangeMax, 
                                 float zRangeMin, float zRangeMax ) : 
                           thePhiRangeMin( phiRangeMin), thePhiRangeMax( phiRangeMax),
                           theZRangeMin( zRangeMin), theZRangeMax( zRangeMax) {
    std::cout << " ConversionBarrelEstimator CTOR " << std::endl;
}

  // zero value indicates incompatible ts - hit pair
  virtual std::pair<bool,double> estimate( const TrajectoryStateOnSurface& ts, 
                               const TransientTrackingRecHit& hit	) const;
  virtual bool  estimate( const TrajectoryStateOnSurface& ts, 
				       const BoundPlane& plane) const;
  virtual ConversionBarrelEstimator* clone() const {
    return new ConversionBarrelEstimator(*this);
  } 





  virtual Local2DVector maximalLocalDisplacement( const TrajectoryStateOnSurface& ts, const BoundPlane& plane) const;




private:

  float thePhiRangeMin;
  float thePhiRangeMax;
  float theZRangeMin;
  float theZRangeMax;




};

#endif // ConversionBarrelEstimator_H
