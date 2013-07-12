// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ForwardMeasurementEstimator
//
/**\class ForwardMeasurementEstimator EgammaElectronAlgos/ForwardMeasurementEstimator

 Description: MeasurementEstimator for Pixel Endcap, ported from ORCA

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ForwardMeasurementEstimator.cc,v 1.21 2010/09/21 17:06:14 chamont Exp $
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalDetRangeRPhi.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
// zero value indicates incompatible ts - hit pair
std::pair<bool,double> ForwardMeasurementEstimator::estimate( const TrajectoryStateOnSurface& ts,
							      const TransientTrackingRecHit& hit) const {
  LocalPoint lp = hit.localPosition();
  GlobalPoint gp = hit.det()->surface().toGlobal( lp);
  return estimate(ts,gp);
}

std::pair<bool,double> ForwardMeasurementEstimator::estimate( const TrajectoryStateOnSurface& ts, GlobalPoint &gp) const {

  float tsR = ts.globalParameters().position().perp();
  float tsPhi = ts.globalParameters().position().phi();

  float rhPhi = gp.phi();
  float rhR = gp.perp();

  float myZ = gp.z();

  float rMin = theRMin;
  float rMax = theRMax;
  float myPhimin = thePhiMin;
  float myPhimax = thePhiMax;

  if(std::abs(myZ)> 70. &&  std::abs(myZ)<170.)
    {
      rMin = theRMinI;
      rMax = theRMaxI;
    }

  float phiDiff = tsPhi - rhPhi;
  if (phiDiff > pi) phiDiff -= twopi;
  if (phiDiff < -pi) phiDiff += twopi;

  float rDiff = tsR - rhR;

  if ( phiDiff < myPhimax && phiDiff > myPhimin &&
       rDiff < rMax && rDiff > rMin) {
    return std::pair<bool,double>(true,1.);
  } else {
    return std::pair<bool,double>(false,0.);
  }
}

std::pair<bool,double> ForwardMeasurementEstimator::estimate
 ( const GlobalPoint& vprim,
   const TrajectoryStateOnSurface& absolute_ts,
   GlobalPoint & absolute_gp ) const
 {
  GlobalVector ts = absolute_ts.globalParameters().position() - vprim ;
  GlobalVector gp = absolute_gp - vprim ;

  float tsR = ts.perp();
  float tsPhi = ts.phi();

  float rhPhi = gp.phi();
  float rhR = gp.perp();

  float myZ = gp.z();

  float rMin = theRMin;
  float rMax = theRMax;
  float myPhimin = thePhiMin;
  float myPhimax = thePhiMax;

  if(std::abs(myZ)> 70. &&  std::abs(myZ)<170.)
    {
      rMin = theRMinI;
      rMax = theRMaxI;
    }

  float phiDiff = normalized_phi(rhPhi - tsPhi) ;
  float rDiff = rhR - tsR;

  if ( phiDiff < myPhimax && phiDiff > myPhimin && rDiff < rMax && rDiff > rMin)
   { return std::pair<bool,double>(true,1.) ; }
  else
   { return std::pair<bool,double>(false,0.) ; }
}

bool ForwardMeasurementEstimator::estimate
 ( const TrajectoryStateOnSurface & ts,
   const BoundPlane & plane ) const
 {
  typedef std::pair<float,float> Range ;

  GlobalPoint trajPos(ts.globalParameters().position());
  GlobalDetRangeRPhi detRange(plane);

  float r1 = 0.;
  float r2 = 40.;

  Range trajRRange(trajPos.perp() - r1, trajPos.perp() + r2);
  Range trajPhiRange(trajPos.phi() - std::abs(thePhiMin), trajPos.phi() + std::abs(thePhiMax));

  if(rangesIntersect(trajRRange, detRange.rRange()) &&
     rangesIntersect(trajPhiRange, detRange.phiRange(), PhiLess()))
   { return true ; }
  else
   { return false ; }
}

MeasurementEstimator::Local2DVector
ForwardMeasurementEstimator::maximalLocalDisplacement
 ( const TrajectoryStateOnSurface & ts,
   const BoundPlane & plane) const
 {
  float nSigmaCut = 3.;
  if ( ts.hasError())
   {
    LocalError le = ts.localError().positionError();
    return Local2DVector( sqrt(le.xx())*nSigmaCut, sqrt(le.yy())*nSigmaCut);
   }
  else
    return Local2DVector(999999,999999) ;
 }



