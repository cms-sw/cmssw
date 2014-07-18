#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionForwardEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"




  // zero value indicates incompatible ts - hit pair
std::pair<bool,double> ConversionForwardEstimator::estimate( const TrajectoryStateOnSurface& ts, 
							const TrackingRecHit& hit) const {
  LogDebug("ConversionForwardEstimator") << "ConversionForwardEstimator::estimate( const TrajectoryStateOnSurface& ts ...) " << "\n";
  //  std::cout  << "ConversionForwardEstimator::estimate( const TrajectoryStateOnSurface& ts ...) " << "\n";
  
  std::pair<bool,double> result;
  
  float tsPhi = ts.globalParameters().position().phi();
  GlobalPoint gp = hit.globalPosition();
  float rhPhi = gp.phi();
  float rhR = gp.perp();

  // allow an r fudge of 1.5 * times the sigma
  // nodt used float dr = 1.5 * hit.localPositionError().yy();
  //std::cout << " err " << hit.globalPositionError().phierr(gp) 
  //    << " "     << hit.globalPositionError().rerr(gp) << std::endl;

  // not used float zLayer = ts.globalParameters().position().z();
  float rLayer = ts.globalParameters().position().perp();

  float newdr = sqrt(pow(dr_,2)+4.*hit.localPositionError().yy());
  float rMin = rLayer - newdr;
  float rMax = rLayer + newdr;
  float phiDiff = tsPhi - rhPhi;
  if (phiDiff > pi) phiDiff -= twopi;
  if (phiDiff < -pi) phiDiff += twopi; 

  //std::cout << " ConversionForwardEstimator: RecHit at " << gp << "\n";
  //std::cout << "                   rMin = " << rMin << ", rMax = " << rMax << ", rHit = " << rhR << "\n";
  //std::cout << "                   thePhiRangeMin = " << thePhiRangeMin << ", thePhiRangeMax = " << thePhiRangeMax << ", phiDiff = " << phiDiff << "\n";

  
  if ( phiDiff < thePhiRangeMax && phiDiff > thePhiRangeMin && 
       rhR < rMax && rhR > rMin) {
  
    
    //    std::cout << "      estimator returns 1 with phiDiff " << thePhiRangeMin << " < " << phiDiff << " < "
    // << thePhiRangeMax << " and rhR " << rMin << " < " << rhR << " < " << rMax << "\n";
    //std::cout << " YES " << phiDiff << " " <<rLayer-rhR << "\n";
    //std::cout << "                  => RECHIT ACCEPTED " << "\n";
   
    result.first= true;
    result.second=phiDiff;
  } else {
    /*
    cout << "      estimator returns 0 with phiDiff " << thePhiRangeMin << " < " << phiDiff << " < "
     << thePhiRangeMax << " and  rhR " << rMin << " < " << rhR << " < " << rMax << endl;
    */
    result.first= false;
    result.second=0;    
    
  }
  
  return result;
  
}

bool ConversionForwardEstimator::estimate( const TrajectoryStateOnSurface& ts, 
			   const BoundPlane& plane) const {

  //  std::cout << "ConversionForwardEstimator::estimate( const TrajectoryStateOnSurface& ts, const BoundPlane& plane) always TRUE " << "\n";  
  // this method should return one if a detector ring is close enough
  //     to the hit, zero otherwise.
  //     Now time is wasted looking for hits in the rings which are anyhow
  //     too far from the prediction   
  return true ;

}



MeasurementEstimator::Local2DVector
ConversionForwardEstimator::maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
                                                        const BoundPlane& plane) const
{
  
  /*
  if ( ts.hasError() ) {
    LocalError le = ts.localError().positionError();
    std::cout << "  ConversionForwardEstimator::maximalLocalDisplacent local error " << sqrt(le.xx()) << " " << sqrt(le.yy()) << " nSigma " << nSigmaCut() << " sqrt(le.xx())*nSigmaCut() " << sqrt(le.xx())*nSigmaCut()  << "  sqrt(le.yy())*nSigmaCut() " <<  sqrt(le.yy())*nSigmaCut() << std::endl;
    return Local2DVector( sqrt(le.xx())*nSigmaCut(), sqrt(le.yy())*nSigmaCut());
    
  }
  else return Local2DVector(99999,99999);
  */

  return Local2DVector(99999,99999);

}


