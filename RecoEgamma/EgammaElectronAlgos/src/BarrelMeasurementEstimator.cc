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
// $Id: BarrelMeasurementEstimator.cc,v 1.3 2007/02/05 17:53:52 uberthon Exp $
//
//

#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalDetRangeZPhi.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "CLHEP/Units/PhysicalConstants.h"


// zero value indicates incompatible ts - hit pair
std::pair<bool,double> BarrelMeasurementEstimator::estimate( const TrajectoryStateOnSurface& ts, 
							     const TransientTrackingRecHit& hit) const {

  float tsPhi = ts.globalParameters().position().phi();
  LocalPoint lp = hit.localPosition();
  GlobalPoint gp = hit.det()->surface().toGlobal( lp); 

  float rhPhi = gp.phi();
  
  float zDiff = ts.globalParameters().position().z() - gp.z(); 
  float phiDiff = tsPhi - rhPhi;
  if (phiDiff > pi) phiDiff -= twopi;
  if (phiDiff < -pi) phiDiff += twopi; 


  if ( phiDiff < thePhiRangeMax && phiDiff > thePhiRangeMin && 
       zDiff < theZRangeMax && zDiff > theZRangeMin) {

    return std::pair<bool,double>(true,1.);
  } else {

    //     cout<<" barrel rechit est returns false!!"<<endl;
    //     cout << " phiDiff,thePhiRangeMax,thePhiRangeMin  "<<phiDiff<<" "<<thePhiRangeMax<<" "<< thePhiRangeMin<<endl;
    //     cout << " zDiff, theZRangeMax, theZRangeMin "<<zDiff<<" "<<theZRangeMax<<" "<< theZRangeMin<<endl;
    return std::pair<bool,double>(false,0.);
  }
}

bool BarrelMeasurementEstimator::estimate( const TrajectoryStateOnSurface& ts, 
					   const BoundPlane& plane) const {
    
  typedef std::pair<float,float> Range; 


  GlobalPoint trajPos(ts.globalParameters().position());
  GlobalDetRangeZPhi detRange(plane);

  Range trajZRange(trajPos.z() - fabs(theZRangeMin), trajPos.z() + fabs(theZRangeMax));
  Range trajPhiRange(trajPos.phi() - fabs(thePhiRangeMin), trajPos.phi() + fabs(thePhiRangeMax));

  if(rangesIntersect(trajZRange, detRange.zRange()) &&
     rangesIntersect(trajPhiRange, detRange.phiRange(), PhiLess())) {
    return true;
  }
  else { 
    //     cout <<cout<<" barrel boundpl est returns false!!"<<endl;
    //     cout<<"BarrelMeasurementEstimator(estimate) :thePhiRangeMin,thePhiRangeMax, theZRangeMin,theZRangeMax "<<thePhiRangeMin<<" "<<thePhiRangeMax<<" "<< theZRangeMin<<" "<<theZRangeMax<<endl;
    //     cout<<" trajZRange "<<trajZRange.first<<" "<<trajZRange.second<<endl;
    //     cout<<" trajPhiRange "<<trajPhiRange.first<<" "<<trajPhiRange.second<<endl;
    //     cout<<" detZRange "<<detRange.zRange().first<<" "<<detRange.zRange().second<<endl;
    //     cout<<" detPhiRange "<<detRange.phiRange().first<<" "<<detRange.phiRange().second<<endl;
    //     cout<<" intersect z: "<<rangesIntersect(trajZRange, detRange.zRange())<<endl;
    //     cout<<" intersect phi: "<<rangesIntersect(trajPhiRange, detRange.phiRange(), PhiLess())<<endl;
    return false;
  }

}

MeasurementEstimator::Local2DVector 
BarrelMeasurementEstimator::maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
							const BoundPlane& plane) const
{
  // completely temporary version
  float nSigmaCut = 3.;
  if ( ts.hasError()) {
    LocalError le = ts.localError().positionError();
    return Local2DVector( sqrt(le.xx())*nSigmaCut, sqrt(le.yy())*nSigmaCut);
  }
  //UB FIXME!!!!!!!  else return Local2DVector(0,0);
  else return Local2DVector(99999,99999);
}



