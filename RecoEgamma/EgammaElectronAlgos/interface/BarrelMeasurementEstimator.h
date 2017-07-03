#ifndef BarrelMeasurementEstimator_H
#define BarrelMeasurementEstimator_H
// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      BarrelMeasurementEstimator
//
/**\class ElectronSeedProducer EgammaElectronAlgos/BarrelMeasurementEstimator

 Description: MeasurementEstimator for Pixel Barrel, ported from ORCA
 Class defining the search area in the barrel in the pixel match
 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include <utility>


class BarrelMeasurementEstimator : public MeasurementEstimator
 {
  public:

    BarrelMeasurementEstimator()
     {}
    BarrelMeasurementEstimator(float phiMin, float phiMax, float zMin, float zMax )
     : thePhiMin(phiMin), thePhiMax(phiMax), theZMin(zMin), theZMax(zMax)
     {}

    void setPhiRange( float dummyphiMin , float dummyphiMax )
     { thePhiMin = dummyphiMin ; thePhiMax = dummyphiMax ; }
    void setZRange( float zmin, float zmax )
     { theZMin=zmin ; theZMax=zmax ; }

    // zero value indicates incompatible ts - hit pair
    std::pair<bool,double> estimate( const TrajectoryStateOnSurface & ts, const TrackingRecHit & hit ) const override ;
    virtual std::pair<bool,double> estimate( const TrajectoryStateOnSurface & ts, const GlobalPoint & gp ) const ;
    virtual std::pair<bool,double> estimate( const GlobalPoint & vprim, const TrajectoryStateOnSurface & ts, const GlobalPoint & gp ) const ;
    bool estimate( const TrajectoryStateOnSurface & ts, const BoundPlane & plane) const override ;

    BarrelMeasurementEstimator* clone() const override
     { return new BarrelMeasurementEstimator(*this) ; }

    MeasurementEstimator::Local2DVector
    maximalLocalDisplacement( const TrajectoryStateOnSurface & ts, const BoundPlane & plane) const override ;

  private:

    float thePhiMin ;
    float thePhiMax ;
    float theZMin ;
    float theZMax ;

 } ;

#endif // BarrelMeasurementEstimator_H
