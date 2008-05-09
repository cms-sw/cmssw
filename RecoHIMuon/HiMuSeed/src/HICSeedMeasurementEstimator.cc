#include "RecoHIMuon/HiMuSeed/interface/HICSeedMeasurementEstimator.h"
#include <CLHEP/Units/PhysicalConstants.h>

namespace cms {
MeasurementEstimator::HitReturnType HICSeedMeasurementEstimator::estimate(const TrajectoryStateOnSurface& ts,
					                         const TransientTrackingRecHit& hit) const
{
  double dfimean = 0.;
  double dzmean = 0.;
    
//
// For mixed are 0.8<|eta|<1.6 there is a shift for + and -
//
  if( fabs(ts.freeTrajectoryState()->parameters().momentum().eta()) > 0.8 && 
      fabs(ts.freeTrajectoryState()->parameters().momentum().eta()) < 1.6 )
  {
     dfimean = -0.14;
     if( ts.freeTrajectoryState()->charge() > 0. ) dfimean = -0.24;
     dzmean = -10.;
  }
  if( fabs(ts.freeTrajectoryState()->parameters().momentum().eta()) > 1.6 )
  {
     dfimean = -0.02;
     if( ts.freeTrajectoryState()->charge() > 0. ) dfimean = -0.1776; 
  }

  if( !(hit.isValid()) ) return HitReturnType(false,0.);
  
  double dfi = fabs( ts.freeTrajectoryState()->parameters().position().phi()-
                                             hit.globalPosition().phi() + dfimean );
  if(dfi>pi) dfi=twopi-dfi;

  if(dfi>thePhi) return HitReturnType(false,0.);

  if( fabs(hit.globalPosition().z()) > 120.)
  {
  
  if(fabs(hit.globalPosition().perp()-ts.freeTrajectoryState()->parameters().position().perp())
                                                                                       >theZ) return HitReturnType(false,0.);
  }
   else
   {
   
      if(fabs(hit.globalPosition().z()-ts.freeTrajectoryState()->parameters().position().z() + dzmean)
                                                                                                    >theZ) return HitReturnType(false,0.);
   }
return HitReturnType(true,1.);  				     
}

MeasurementEstimator::SurfaceReturnType HICSeedMeasurementEstimator::estimate(const TrajectoryStateOnSurface& ts,
					                const BoundPlane& plane) const
{
   double length = plane.bounds().length();
   double width = plane.bounds().width();
   double deltafi = atan2( width , (double)plane.position().perp() );
   double phitrack = ts.freeTrajectoryState()->parameters().position().phi();
   double phiplane = plane.position().phi();

   if( phitrack < 0. ) phitrack = twopi + phitrack;
   if( phiplane < 0. ) phiplane = twopi + phiplane;
   
   double dphi = fabs(phitrack - phiplane);

   if( dphi > pi ) dphi = twopi - dphi;
   if( dphi >  thePhi + deltafi ) {
     
     return false;
   }
   if ( fabs( plane.position().z() ) < 111. )
   {
     // barrel
        if( fabs(ts.freeTrajectoryState()->parameters().position().z() - plane.position().z()) 
   	                                                               >  theZ+length ) {
								       return false;
								       }
   } 
     else
     {
      // forward
        if( fabs(ts.freeTrajectoryState()->parameters().position().perp() - plane.position().perp()) 
   	                                                                            >  theZ + length ) {
										    return false;
										    }
     }
   return true;					 
}

MeasurementEstimator::Local2DVector 
HICSeedMeasurementEstimator::maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
							const BoundPlane& plane) const
{
  if ( ts.hasError()) {
    LocalError le = ts.localError().positionError();
    return Local2DVector( sqrt(le.xx())*nSigmaCut(), sqrt(le.yy())*nSigmaCut());
  }
  else return Local2DVector(0,0);
}
}
