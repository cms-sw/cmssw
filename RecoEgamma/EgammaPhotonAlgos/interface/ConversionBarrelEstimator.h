#ifndef RecoEGAMMA_ConversionBarrelEstimator_H
#define RecoEGAMMA_ConversionBarrelEstimator_H
/**
 * \class ConversionBarrelEstimator
 *  Defines the search area in the barrel 
 *
 *   $Date: 2012/05/29 08:23:53 $
 *   $Revision: 1.5 $
 *   \author Nancy Marinelli, U. of Notre Dame, US
 */

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h" 
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"


class TrajectoryStateOnSurface;
class RecHit;
class Plane;

class ConversionBarrelEstimator : public MeasurementEstimator {
public:

  ConversionBarrelEstimator() {};
  ConversionBarrelEstimator( float phiRangeMin, float phiRangeMax, 
                                 float zRangeMin, float zRangeMax, double nSigma = 3. ) : 
                           thePhiRangeMin( phiRangeMin), thePhiRangeMax( phiRangeMax),
                           theZRangeMin( zRangeMin), theZRangeMax( zRangeMax), theNSigma(nSigma) {
    //    std::cout << " ConversionBarrelEstimator CTOR " << std::endl;
}

  // zero value indicates incompatible ts - hit pair
  virtual std::pair<bool,double> estimate( const TrajectoryStateOnSurface& ts, 
                               const TransientTrackingRecHit& hit	) const;
  virtual bool  estimate( const TrajectoryStateOnSurface& ts, 
				       const Plane& plane) const;
  virtual ConversionBarrelEstimator* clone() const {
    return new ConversionBarrelEstimator(*this);
  } 





  virtual Local2DVector maximalLocalDisplacement( const TrajectoryStateOnSurface& ts, const BoundPlane& plane) const;


  double nSigmaCut() const {return theNSigma;}

private:

  float thePhiRangeMin;
  float thePhiRangeMax;
  float theZRangeMin;
  float theZRangeMax;
  double theNSigma;



};

#endif // ConversionBarrelEstimator_H
