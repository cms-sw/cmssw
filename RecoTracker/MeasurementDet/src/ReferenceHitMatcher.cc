#include "RecoTracker/MeasurementDet/interface/ReferenceHitMatcher.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"

#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"

using namespace std;
#include <iostream>

ReferenceHitMatcher::RecHitContainer 
ReferenceHitMatcher::match( const RecHitContainer& monoHits,
			const RecHitContainer& stereoHits,
			const GluedGeomDet& gdet,
			const LocalVector& dir) const
{
  RecHitContainer result;
  for (RecHitContainer::const_iterator mh=monoHits.begin(); mh!=monoHits.end(); mh++) {
    for (RecHitContainer::const_iterator sh=stereoHits.begin(); sh!=stereoHits.end(); sh++) {
      ReturnType matched = match( **mh, **sh, gdet, dir);
      if (matched.first) result.push_back( matched.second);
    }
  }
  return result;
}

ReferenceHitMatcher::ReturnType 
ReferenceHitMatcher::match( const TransientTrackingRecHit& monoHit, 
			const TransientTrackingRecHit& stereoHit,
			const GluedGeomDet& gdet,
			const LocalVector& dir) const
{
  const BoundPlane& plane = gdet.surface();
  std::pair<LocalPoint,LocalVector> monoPair = 
    projectHit( monoHit, plane, dir);
  std::pair<LocalPoint,LocalVector> stereoPair = 
    projectHit( stereoHit, plane, dir);

  LocalPoint crossingPoint = crossing( monoPair, stereoPair);
  if ( plane.bounds().inside(crossingPoint)) {
    // the two hits match
    // LocalError monoErr   = rotateError( monoHit, gdet.surface());
    // LocalError stereoErr = rotateError( stereoHit, gdet.surface());
    //LocalError matchedError = weightedMean( monoErr, stereoErr);

    // The weightedMean does not give correct results in trapezoidal detectors yet (to be debuged),
    // so the ORCA implementation is used instead
    LocalError matchedError = orcaMatchedError( monoHit, stereoHit, gdet);

    // RecHit matchedHit( new MatchedRecHit( gdet, monoHit, stereoHit, crossingPoint, matchedError));

    SiStripRecHit2DMatchedLocalPos* hitData = 
      new SiStripRecHit2DMatchedLocalPos( crossingPoint, matchedError, 
					  gdet.geographicalId(),
					  dynamic_cast<const SiStripRecHit2DLocalPos*>(monoHit.hit()), 
					  dynamic_cast<const SiStripRecHit2DLocalPos*>(stereoHit.hit()));
    TSiStripMatchedRecHit* matchedHit = new TSiStripMatchedRecHit( &gdet, hitData);

    return ReturnType( true, matchedHit);
  }
  else {
    return ReturnType( false, 0);
  }
}


LocalError ReferenceHitMatcher::weightedMean( const LocalError& a, const LocalError& b) const
{
  double d1 = a.xx()*a.yy() -  a.xy()*a.xy();
  double d2 = b.xx()*b.yy() -  b.xy()*b.xy();

  double exx =  d2*a.yy() + d1*b.yy();
  double exy = -d2*a.xy() - d1*b.xy();
  double eyy =  d2*a.xx() + d1*b.xx();

  double det = d1*d2 / (exx*eyy - exy*exy);

  return LocalError( det*eyy, -det*exy, det*exx);
}

std::pair<LocalPoint,LocalVector> ReferenceHitMatcher::projectHit( const TransientTrackingRecHit& hit, 
							       const BoundPlane& plane,
							       const LocalVector& dir) const
{
  const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(hit.det());
  if (stripDet == 0) throw MeasurementDetException("HitMatcher hit is not on StripGeomDetUnit");

  const StripTopology& topol = stripDet->specificTopology();

  LocalPoint localHit = plane.toLocal(hit.globalPosition());

  float scale = -localHit.z() / dir.z(); 

  LocalPoint projectedPos = localHit + scale*dir;

  float selfAngle = -topol.stripAngle( topol.strip( hit.localPosition()));
  
  LocalVector stripDir( sin(selfAngle), cos(selfAngle), 0); // vector along strip in hit frame

  LocalVector localStripDir( plane.toLocal( hit.det()->surface().toGlobal( stripDir)));

  return std::pair<LocalPoint,LocalVector>( projectedPos, localStripDir);
}

LocalPoint ReferenceHitMatcher::crossing( const std::pair<LocalPoint,LocalVector>& a,
				      const std::pair<LocalPoint,LocalVector>& b) const
{
  const LocalPoint& p1( a.first);
  const LocalVector& v1( a.second);
  const LocalPoint& p2( b.first);
  const LocalVector& v2( b.second);

  double den = v1.x()*v2.y() - v2.x()*v1.y();

  if (den == 0) return LocalPoint( 1.e10, 1.e10, 0);

  double beta = v1.x()*(p1.y()-p2.y())/den + v1.y()*(p2.x()-p1.x())/den;
  //  double alpha = (p2.x() + beta*v2.x() - p1.x()) / v1.x();

  //  LocalPoint cross1 = p1 + alpha*v1;
  LocalPoint cross2 = p2 + beta*v2;

  return cross2;
}

LocalError ReferenceHitMatcher::rotateError( const TransientTrackingRecHit& hit,
					 const Plane& plane) const
{
  LocalVector hitXaxis = plane.toLocal( hit.det()->surface().toGlobal( LocalVector(1,0,0)));

  LocalError rotatedError = hit.localPositionError().rotate( hitXaxis.x(), hitXaxis.y());

  return rotatedError;
}

void ReferenceHitMatcher::dumpHit(const TransientTrackingRecHit& hit) const
{
  if (hit.isValid()) {
    GlobalPoint gp = hit.globalPosition();
    LocalError le = hit.localPositionError();
    edm::LogInfo("MeasurementDet") << "Hit position (r,phi,z) (" 
				   << gp.perp() << ", "
				   << gp.phi() << ", "
				   << gp.z() << ") localPos "
				   << hit.localPosition() << " localErr ("
				   << sqrt(le.xx()) << ", "
				   << le.xy() << ", "
				   << sqrt(le.yy()) << ")" ;
  }
  else {
    const GeomDet* det = hit.det();
    if (det != 0) {
      GlobalPoint gp = det->position();
      edm::LogInfo("MeasurementDet") << "Invalid hit in det at position (r,phi,z) ("
				     << gp.perp() << ", "
				     << gp.phi() << ", "
				     << gp.z() ;
    }
    else {
       edm::LogInfo("MeasurementDet") << "Invalid hit on DetLayer" ;     
    }
  }
}

LocalVector ReferenceHitMatcher::dloc2( const LocalError& err) const
{
  double q = 2.*err.xy()/(err.xx() - err.yy());
  double t = q/(1. + sqrt(1.+q*q));
  return LocalVector(-t/sqrt(1.+t*t), 1./sqrt(1.+t*t), 0.);
}

double ReferenceHitMatcher::sigp2( const LocalError& err) const
{
  double vvpww = err.xx() + err.yy();
  double vvmww = err.xx() - err.yy();
  double sigl22 = .5*(vvpww + sqrt(vvmww*vvmww + 4.*err.xy()*err.xy()));
  return (err.xx()*err.yy() - err.xy()*err.xy())/sigl22;
}

LocalError ReferenceHitMatcher::orcaMatchedError( const TransientTrackingRecHit& monoHit, 
					      const TransientTrackingRecHit& stereoHit,
					      const GluedGeomDet& gdet) const
  //const LocalError& monoErr, const LocalError& stereoErr) const
{
  GlobalVector d1 = monoHit.det()->surface().toGlobal( dloc2(monoHit.localPositionError()));
  LocalVector d1atGlued = gdet.surface().toLocal(d1);

  GlobalVector d2 = stereoHit.det()->surface().toGlobal( dloc2(stereoHit.localPositionError()));
  LocalVector d2atGlued = gdet.surface().toLocal(d2);

  double s1 = -d1atGlued.x(); double c1 = d1atGlued.y();
  double s2 = -d2atGlued.x(); double c2 = d2atGlued.y();
  double q2 = c1*s2 - c2*s1; q2 *= q2;

  double sigp21 = sigp2( monoHit.localPositionError());
  double sigp22 = sigp2( stereoHit.localPositionError());

  double errvv = (sigp21*s2*s2 + sigp22*s1*s1)/q2;
  double errvw =-(sigp21*c2*s2 + sigp22*c1*s1)/q2;
  double errww = (sigp21*c2*c2 + sigp22*c1*c1)/q2;

  return LocalError(errvv, errvw, errww);
}
