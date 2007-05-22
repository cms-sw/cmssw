#ifndef ForwardMeasurementEstimator_H
#define ForwardMeasurementEstimator_H

// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ForwardMeasurementEstimator
// 
/**\class ForwardMeasurementEstimator EgammaElectronAlgos/ForwardMeasurementEstimator

 Description: Class defining the search area in the forward disks in the pixel match, ported from ORCA

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ForwardMeasurementEstimator.h,v 1.3 2007/03/08 18:34:11 futyand Exp $
//
//
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include <utility>

class ForwardMeasurementEstimator : public MeasurementEstimator {
public:
  ForwardMeasurementEstimator() {};
  ForwardMeasurementEstimator( float phiRangeMin, float phiRangeMax, 
                                 float zRangeMin, float zRangeMax ) : 
                           thePhiRangeMin( phiRangeMin), thePhiRangeMax( phiRangeMax),
                           theZRangeMin( zRangeMin), theZRangeMax( zRangeMax) {}

  void setPhiRange (float dummyphiRangeMin , float dummyphiRangeMax) 
  { 
    thePhiRangeMin = dummyphiRangeMin ; 
    thePhiRangeMax = dummyphiRangeMax ; 
  }
  
  // zero value indicates incompatible ts - hit pair
  virtual std::pair<bool,double> estimate( const TrajectoryStateOnSurface& ts, 
			   const TransientTrackingRecHit& hit) const;
  virtual bool estimate( const TrajectoryStateOnSurface& ts, 
			   const BoundPlane& plane) const;

  virtual ForwardMeasurementEstimator* clone() const
    {
      return new ForwardMeasurementEstimator(*this);
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

#endif // ForwardMeasurementEstimator_H
