#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionBarrelEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h" 
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"


  // zero value indicates incompatible ts - hit pair
std::pair<bool,double> ConversionBarrelEstimator::estimate( const TrajectoryStateOnSurface& ts, 
							    const TrackingRecHit& hit) const {
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
  std::cout << "  BarrelEstimator ts local error " <<ts.localError().positionError()  << " hit local error " << hit.localPositionError() << std::endl; 
  std::cout << "  BarrelEstimator:  RecHit at " << gp << " phi " << rhPhi << " eta " << gp.eta() <<  std::endl;
  std::cout << "  BarrelEstimator:  ts at " << ts.globalParameters().position() << " phi " <<ts.globalParameters().position().phi() << " eta " << ts.globalParameters().position().eta()<<  std::endl;
  std::cout << "                    zrange = +/-" << zrange << ", zDiff = " << zDiff << std::endl;
  std::cout << "                    thePhiRangeMin = " << thePhiRangeMin << ", thePhiRangeMax = " << thePhiRangeMax << ", phiDiff = " << phiDiff << std::endl;
  */

  
  
  if ( phiDiff < thePhiRangeMax && phiDiff > thePhiRangeMin && 
       zDiff < zrange && zDiff > -zrange) {
    
    //    std::cout << "      estimator returns 1 with phiDiff " << thePhiRangeMin << " < " << phiDiff << " < "
    //    << thePhiRangeMax << " and zDiff " << zDiff << " < " << zrange << std::endl;
    // std::cout << " YES " << phiDiff << " " << zDiff << std::endl;
    // std::cout << "                  => RECHIT ACCEPTED " << std::endl;
    
    result.first=true;
    result.second=phiDiff;
  } else {
    
    //     std::cout << "      estimator returns NOT ACCEPTED  with phiDiff " << thePhiRangeMin << " < " << phiDiff << " < "
    //<< thePhiRangeMax << " and zDiff " << zDiff << " < " << theZRangeMax+dz << std::endl;
    
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
  Range trajZRange(trajPos.z() - 2.*theZRangeMax, trajPos.z() + 2.*theZRangeMax);
  Range trajPhiRange(trajPos.phi() + thePhiRangeMin, trajPos.phi() + thePhiRangeMax);


   if(rangesIntersect(trajZRange, plane.zSpan()) &&
      rangesIntersect(trajPhiRange, plane.phiSpan(), [](auto x, auto y){return Geom::phiLess(x, y);})) {
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
  

  
  /* 
  if ( ts.hasError() ) {
    LocalError le = ts.localError().positionError();
    std::cout << "  ConversionBarrelEstimator::maximalLocalDisplacent local error " << sqrt(le.xx()) << " " << sqrt(le.yy()) << " nSigma " << nSigmaCut() << " sqrt(le.xx())*nSigmaCut() " << sqrt(le.xx())*nSigmaCut()  << "  sqrt(le.yy())*nSigmaCut() " <<  sqrt(le.yy())*nSigmaCut() << std::endl;
    return Local2DVector( sqrt(le.xx())*nSigmaCut(), sqrt(le.yy())*nSigmaCut());
  }

  else return Local2DVector(9999,9999);
  */
return Local2DVector(9999,9999);

}


