#ifndef BarrelMeasurementEstimator_H
#define BarrelMeasurementEstimator_H
// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      BarrelMeasurementEstimator
// 
/**\class ElectronPixelSeedProducer EgammaElectronAlgos/BarrelMeasurementEstimator

 Description: MeasurementEstimator for Pixel Barrel, ported from ORCA
 Class defining the search area in the barrel in the pixel match
 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: BarrelMeasurementEstimator.h,v 1.4 2007/03/08 18:34:11 futyand Exp $
//
//

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include <utility>


class BarrelMeasurementEstimator : public MeasurementEstimator {
public:
  BarrelMeasurementEstimator() {};
  BarrelMeasurementEstimator( float phiRangeMin, float phiRangeMax, 
                              float zRangeMin, float zRangeMax ) : 
                           thePhiRangeMin( phiRangeMin), thePhiRangeMax( phiRangeMax),
                           theZRangeMin( zRangeMin), theZRangeMax( zRangeMax) { }

  void setPhiRange (float dummyphiRangeMin , float dummyphiRangeMax) 
  { 
    thePhiRangeMin = dummyphiRangeMin ; 
    thePhiRangeMax = dummyphiRangeMax ; 
  }
  
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
