#include "CLHEP/Units/PhysicalConstants.h"

//#include "Geometry/Surface/interface/LocalError.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionBarrelEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h" 
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalDetRangeZPhi.h"


  // zero value indicates incompatible ts - hit pair
std::pair<bool,double> ConversionBarrelEstimator::estimate( const TrajectoryStateOnSurface& ts, 
						       const TransientTrackingRecHit& hit) const {
  std::pair<bool,double> result;
  
  //std::cout << "  ConversionBarrelEstimator::estimate( const TrajectoryStateOnSurface& ts, const TransientTrackingRecHit& hit) " << std::endl;
 
  float tsPhi = ts.globalParameters().position().phi();
  GlobalPoint gp = hit.globalPosition();
  float rhPhi = gp.phi();
  
  // allow a z fudge of 2 sigma
  float dz = 2. * sqrt(hit.localPositionError().yy()) ;
  float zDiff = ts.globalParameters().position().z() - gp.z(); 
  float phiDiff = tsPhi - rhPhi;
  if (phiDiff > pi) phiDiff -= twopi;
  if (phiDiff < -pi) phiDiff += twopi; 
  
  // add the errors on the window and the point in quadrature
  float zrange = sqrt(theZRangeMax*theZRangeMax + dz*dz);
  
  /*  
  cout << "  BarrelEstimator:  RecHit at " << gp << endl;
  cout << "                    zrange = +/-" << zrange << ", zDiff = " << zDiff << endl;
  cout << "                    thePhiRangeMin = " << thePhiRangeMin << ", thePhiRangeMax = " << thePhiRangeMax << ", phiDiff = " << phiDiff << endl;
  
  */
  
  if ( phiDiff < thePhiRangeMax && phiDiff > thePhiRangeMin && 
       zDiff < zrange && zDiff > -zrange) {
    /*    
    cout << "      estimator returns 1 with phiDiff " << thePhiRangeMin << " < " << phiDiff << " < "
         << thePhiRangeMax << " and zDiff " << zDiff << " < " << zrange << endl;
    cout << " YES " << phiDiff << " " << zDiff << endl;
    cout << "                  => RECHIT ACCEPTED " << endl;
    */
    result.first=true;
    result.second=phiDiff;
  } else {
    /*
     cout << "      estimator returns 0 with phiDiff " << thePhiRangeMin << " < " << phiDiff << " < "
     << thePhiRangeMax << " and zDiff " << zDiff << " < " << theZRangeMax+dz << endl;
    */
    result.first=false;
    result.second=0;
    
  }
  
  return result;
  
}



bool ConversionBarrelEstimator::estimate( const TrajectoryStateOnSurface& ts, 
						       const BoundPlane& plane) const {
  
  typedef     std::pair<float,float>   Range;
  //  std::cout << "  ConversionBarrelEstimator::estimate( const TrajectoryStateOnSurface& ts, const BoundPlane& plane) " << std::endl;

  GlobalPoint trajPos(ts.globalParameters().position());
  GlobalDetRangeZPhi detRange(plane);
  Range trajZRange(trajPos.z() - 2.*theZRangeMax, trajPos.z() + 2.*theZRangeMax);
  Range trajPhiRange(trajPos.phi() + thePhiRangeMin, trajPos.phi() + thePhiRangeMax);


   if(rangesIntersect(trajZRange, detRange.zRange()) &&
      rangesIntersect(trajPhiRange, detRange.phiRange(), PhiLess())) {
     //     std::cout << "   ConversionBarrelEstimator::estimate( const TrajectoryStateOnSurface& ts, const BoundPlane& plane)  IN RANGE " << std::endl;  
    return true;   



  } else {

    //    std::cout << "   ConversionBarrelEstimator::estimate( const TrajectoryStateOnSurface& ts, const BoundPlane& plane) NOT IN RANGE " << std::endl;  
    return false;

  }


}


MeasurementEstimator::Local2DVector
ConversionBarrelEstimator::maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
                                                        const BoundPlane& plane) const
{
  

  
  
  if ( ts.hasError() ) {
    LocalError le = ts.localError().positionError();
    //    std::cout << "  ConversionBarrelEstimator::maximalLocalDisplacent local error " << le.xx() << " " << le.yy() << std::endl;
    return Local2DVector( sqrt(le.xx())*nSigmaCut(), sqrt(le.yy())*nSigmaCut());
    //    return Local2DVector( sqrt(le.xx()), sqrt(le.yy()) );
  }
  else return Local2DVector(0,0);
 

}


