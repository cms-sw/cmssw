#ifndef BarrelMeasurementEstimator_H
#define BarrelMeasurementEstimator_H
// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      BarrelMeasurementEstimator
// 
/**\class ElectronPixelSeedProducer EgammaElectronAlgos/BarrelMeasurementEstimator

 Description: MeasurementEstimator for Pixel Barrel, ported from ORCA

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: BarrelMeasurementEstimator.h,v 1.1 2006/06/02 16:21:02 uberthon Exp $
//
//

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"

#include "Geometry/Surface/interface/BoundPlane.h"

/** Class defining the search area in the barrel in the pixel match 
 */

class BarrelMeasurementEstimator : public MeasurementEstimator {
public:
  BarrelMeasurementEstimator() {};
  BarrelMeasurementEstimator( float phiRangeMin, float phiRangeMax, 
                                 float zRangeMin, float zRangeMax ) : 
                           thePhiRangeMin( phiRangeMin), thePhiRangeMax( phiRangeMax),
                           theZRangeMin( zRangeMin), theZRangeMax( zRangeMax) { }

  // zero value indicates incompatible ts - hit pair
  virtual std::pair<bool,double> estimate( const TrajectoryStateOnSurface& ts, 
			   const TransientTrackingRecHit& hit) const;
				      //			   const RecHit& hit) const;
  virtual bool estimate( const TrajectoryStateOnSurface& ts, 
			   const BoundPlane& plane) const;

  virtual BarrelMeasurementEstimator* clone() const
    {
      return new BarrelMeasurementEstimator(*this);
    }
MeasurementEstimator::Local2DVector 
maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
							    const BoundPlane& plane) const;

private:

  float thePhiRangeMin;
  float thePhiRangeMax;
  float theZRangeMin;
  float theZRangeMax;

};

#endif // BarrelMeasurementEstimator_H
