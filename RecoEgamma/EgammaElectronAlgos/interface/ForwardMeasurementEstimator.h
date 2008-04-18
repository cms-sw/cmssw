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
// $Id: ForwardMeasurementEstimator.h,v 1.10 2008/02/28 17:52:36 uberthon Exp $
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
  void setZRange (float zmin , float zmax) 
  { 
    theZRangeMin=zmin;
    theZRangeMax=zmax;
  }
   void setRRange (float rmin , float rmax) 
  { 
    rMin_= rmin;
    rMax_ = rmax;
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
  float rMin_, rMax_;

};

#endif // ForwardMeasurementEstimator_H
