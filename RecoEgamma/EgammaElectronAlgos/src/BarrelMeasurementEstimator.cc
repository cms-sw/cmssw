// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      BarrelMeasurementEstimator
//
/**\class ElectronSeedProducer EgammaElectronAlgos/BarrelMeasurementEstimator

 Description: MeasurementEstimator for Pixel Barrel, ported from ORCA

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalDetRangeZPhi.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"


// zero value indicates incompatible ts - hit pair
std::pair<bool,double> BarrelMeasurementEstimator::estimate( const TrajectoryStateOnSurface& ts,
							     const TrackingRecHit& hit) const {
  LocalPoint lp = hit.localPosition();
  GlobalPoint gp = hit.det()->surface().toGlobal( lp);
  return this->estimate(ts,gp);
}

//usable in case we have no TransientTrackingRecHit
std::pair<bool,double> BarrelMeasurementEstimator::estimate( const TrajectoryStateOnSurface& ts,
					   const GlobalPoint &gp) const {
  
  float myZ = gp.z();
  float myR = gp.perp();
  float zDiff = ts.globalParameters().position().z() - myZ;

  float myZmax =  theZMax;
  float myZmin =  theZMin;

  if(std::abs(myZ)<30. && myR>8.) {
    myZmax = 0.09;
    myZmin = -0.09;
  }

  if( zDiff >= myZmax || zDiff <= myZmin ) return std::pair<bool,double>(false,0.);

  float tsPhi = ts.globalParameters().position().phi();  

  float rhPhi = gp.phi();
  
  float phiDiff = tsPhi - rhPhi;
  if (phiDiff > pi) phiDiff -= twopi;
  if (phiDiff < -pi) phiDiff += twopi;

  if ( phiDiff < thePhiMax && phiDiff > thePhiMin ) {

    return std::pair<bool,double>(true,1.);
     } else {

    return std::pair<bool,double>(false,0.);
    }
}


std::pair<bool,double> BarrelMeasurementEstimator::estimate
 ( const GlobalPoint & vprim,
   const TrajectoryStateOnSurface & absolute_ts,
   const GlobalPoint& absolute_gp ) const
 {
  GlobalVector ts = absolute_ts.globalParameters().position() - vprim ;
  GlobalVector gp = absolute_gp - vprim ;
  
  float myR = gp.perp();
  float myZ = gp.z();
  float zDiff = myZ -ts.z() ;  
  float myZmax =  theZMax;
  float myZmin =  theZMin;
  if(std::abs(myZ)<30. && myR>8.)
    {
      myZmax = 0.09;
      myZmin = -0.09;
    }
  

  if( zDiff >= myZmax || zDiff <= myZmin ) return std::pair<bool,double>(false,0.);

  float rhPhi = gp.phi() ;
  float tsPhi = ts.phi();  
  float phiDiff = normalized_phi(rhPhi-tsPhi) ;

  if ( phiDiff < thePhiMax && phiDiff > thePhiMin )
   { return std::pair<bool,double>(true,1.) ; }
  else
   { return std::pair<bool,double>(false,0.) ; }
 }

bool BarrelMeasurementEstimator::estimate
 ( const TrajectoryStateOnSurface & ts,
   const BoundPlane& plane) const
 {
  typedef std::pair<float,float> Range;


  GlobalPoint trajPos(ts.globalParameters().position());
  GlobalDetRangeZPhi detRange(plane);

  Range trajZRange(trajPos.z() - std::abs(theZMin), trajPos.z() + std::abs(theZMax));
  Range trajPhiRange(trajPos.phi() - std::abs(thePhiMin), trajPos.phi() + std::abs(thePhiMax));

  if(rangesIntersect(trajZRange, detRange.zRange()) &&
     rangesIntersect(trajPhiRange, detRange.phiRange(), PhiLess()))
   {
    return true;
   }
  else
   {
    //     cout <<cout<<" barrel boundpl est returns false!!"<<endl;
    //     cout<<"BarrelMeasurementEstimator(estimate) :thePhiMin,thePhiMax, theZMin,theZMax "<<thePhiMin<<" "<<thePhiMax<<" "<< theZMin<<" "<<theZMax<<endl;
    //     cout<<" trajZRange "<<trajZRange.first<<" "<<trajZRange.second<<endl;
    //     cout<<" trajPhiRange "<<trajPhiRange.first<<" "<<trajPhiRange.second<<endl;
    //     cout<<" detZRange "<<detRange.zRange().first<<" "<<detRange.zRange().second<<endl;
    //     cout<<" detPhiRange "<<detRange.phiRange().first<<" "<<detRange.phiRange().second<<endl;
    //     cout<<" intersect z: "<<rangesIntersect(trajZRange, detRange.zRange())<<endl;
    //     cout<<" intersect phi: "<<rangesIntersect(trajPhiRange, detRange.phiRange(), PhiLess())<<endl;
    return false ;
   }

 }

MeasurementEstimator::Local2DVector
BarrelMeasurementEstimator::maximalLocalDisplacement
 ( const TrajectoryStateOnSurface & ts,
   const BoundPlane & plane) const
 {
  float nSigmaCut = 3. ;
  if ( ts.hasError())
   {
    LocalError le = ts.localError().positionError() ;
    return Local2DVector( sqrt(le.xx())*nSigmaCut, sqrt(le.yy())*nSigmaCut) ;
   }
  else return Local2DVector(99999,99999) ;
 }



