#include "RecoHI/HiMuonAlgos/interface/HICSeedMeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
//#include "CLHEP/Units/GlobalPhysicalConstants.h"

//#define DEBUG

namespace cms {
MeasurementEstimator::HitReturnType HICSeedMeasurementEstimator::estimate(const TrajectoryStateOnSurface& ts,
					                         const TransientTrackingRecHit& hit) const
{
  double dfimean = 0.;
  double dzmean = 0.;
  double pi=4.*atan(1.);
  double twopi=8.*atan(1.);
#ifdef DEBUG
   std::cout<<"  HICSeedMeasurementEstimator::estimate::start::eta "<<ts.freeTrajectoryState()->parameters().momentum().eta() <<std::endl;
#endif
    
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

  if( !(hit.isValid()) ) {
#ifdef DEBUG
   std::cout<<"  HICSeedMeasurementEstimator::estimate::hit is not valid " <<std::endl;
#endif
    return HitReturnType(false,0.);
  }   

#ifdef DEBUG
   std::cout<<"  HICSeedMeasurementEstimator::estimate::hit is valid " <<std::endl;
#endif


  double dfi = fabs( ts.freeTrajectoryState()->parameters().position().phi()-
                                             hit.globalPosition().phi() + dfimean );
  if(dfi>pi) dfi=twopi-dfi;

  if(dfi>thePhi) {
#ifdef DEBUG
   std::cout<<"HICSeedMeasurementEstimator::estimate::failed::phi hit::phi fts::thePhi "<<hit.globalPosition().phi()<<" "<<
   ts.freeTrajectoryState()->parameters().position().phi()<<" "<<dfi<<
   " "<<thePhi<<std::endl;
#endif

   return HitReturnType(false,0.);
  }
  if( fabs(hit.globalPosition().z()) > 120.)
  {
  
  if(fabs(hit.globalPosition().perp()-ts.freeTrajectoryState()->parameters().position().perp())
                                                                                       >theZ) {
#ifdef DEBUG
   std::cout<<"HICSeedMeasurementEstimator::estimate::failed::r hit::r fts::theZ "<<hit.globalPosition().perp()<<" "<<
   ts.freeTrajectoryState()->parameters().position().perp()<<
   " "<<theZ<<std::endl;
#endif   
     return HitReturnType(false,0.);
   }
  }
   else
   {
   
      if(fabs(hit.globalPosition().z()-ts.freeTrajectoryState()->parameters().position().z() + dzmean)
                                                                                                    >theZ) {
#ifdef DEBUG
   std::cout<<"HICSeedMeasurementEstimator::estimate::failed::z hit::z fts::theZ "<<hit.globalPosition().z()<<" "<<
   ts.freeTrajectoryState()->parameters().position().z()<<
   " "<<theZ<<std::endl;
#endif

      return HitReturnType(false,0.);
     }
   }
#ifdef DEBUG
   std::cout<<"HICSeedMeasurementEstimator::estimate::accepted::phi::r::z "<<hit.globalPosition().phi()<<" "<<
                                                                             hit.globalPosition().perp()<<" "<<
                                                                             hit.globalPosition().z()<<
   std::endl;
#endif

return HitReturnType(true,1.);  				     
}

MeasurementEstimator::SurfaceReturnType HICSeedMeasurementEstimator::estimate(const TrajectoryStateOnSurface& ts,
					                const BoundPlane& plane) const
{

  double pi=4.*atan(1.);
  double twopi=8.*atan(1.);

#ifdef DEBUG
   std::cout<<"HICSeedMeasurementEstimator::estimate::Det::start::r,phi,z "<<plane.position().perp()<<" "<<plane.position().phi()<<" "<<
   plane.position().z()<<std::endl;
   std::cout<<"HICSeedMeasurementEstimator::estimate::Track::r,phi,z "<<ts.freeTrajectoryState()->parameters().position().perp()<<" "<<
                                                                        ts.freeTrajectoryState()->parameters().position().phi()<<" "<<
                                                                        ts.freeTrajectoryState()->parameters().position().z()<<std::endl;
#endif
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
#ifdef DEBUG
   std::cout<<"HICSeedMeasurementEstimator::estimate::Det::phi failed "<<dphi<<" "<<thePhi<<" "<<deltafi<<std::endl;
#endif     
     return false;
   }
   if ( fabs( plane.position().z() ) < 111. )
   {
     // barrel
        if( fabs(ts.freeTrajectoryState()->parameters().position().z() - plane.position().z()) 
   	                                                               >  theZ+length ) {
      
#ifdef DEBUG
   double dz = fabs(ts.freeTrajectoryState()->parameters().position().z() - plane.position().z());
   std::cout<<"HICSeedMeasurementEstimator::estimate::Det::z failed "<<dz<<" "<<theZ<<" "<<length<<std::endl;
#endif

								       return false;
								       }
   } 
     else
     {
      // forward
        if( fabs(ts.freeTrajectoryState()->parameters().position().perp() - plane.position().perp()) 
   	                                                                            >  theZ + length ) {
#ifdef DEBUG
   double dr = fabs(ts.freeTrajectoryState()->parameters().position().perp() - plane.position().perp());
   std::cout<<"HICSeedMeasurementEstimator::estimate::Det::r failed "<<dr<<" "<<theZ<<" "<<length<<std::endl;
#endif

										    return false;
										    }
     }
#ifdef DEBUG
     std::cout<<" Good detector "<<std::endl;
#endif
   return true;					 
}

MeasurementEstimator::Local2DVector 
HICSeedMeasurementEstimator::maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
							const BoundPlane& plane) const
{
#ifdef DEBUG
   std::cout<<"HICSeedMeasurementEstimator::malLocalDisplacement::start"
   <<ts.hasError()<<std::endl;
#endif

  if ( ts.hasError()) {
    LocalError le = ts.localError().positionError();
#ifdef DEBUG
   std::cout<<"HICSeedMeasurementEstimator::malLocalDisplacement::localerror::sigma"<<Local2DVector( sqrt(le.xx())*nSigmaCut(), sqrt(le.yy())*nSigmaCut())<<std::endl;
#endif    
    return Local2DVector( sqrt(le.xx())*nSigmaCut(), sqrt(le.yy())*nSigmaCut());
  }
  else return Local2DVector(0,0);
}
}
