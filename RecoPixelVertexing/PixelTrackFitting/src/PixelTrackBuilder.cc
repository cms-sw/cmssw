#include "PixelTrackBuilder.h"


#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


template <class T> T sqr( T t) {return t*t;}

reco::Track * PixelTrackBuilder::build(
      const Measurement1D & pt,
      const Measurement1D & phi, 
      const Measurement1D & cotTheta,
      const Measurement1D & tip,  
      const Measurement1D & zip,
      float chi2,
      int   charge,
      const std::vector<const TrackingRecHit* >& hits,
      const MagneticField * mf) const 
{

  LogDebug("PixelTrackBuilder::build")
    <<" reconstructed TRIPLET kinematics: " 
    <<" \t pt: " << pt.value() <<"+/-"<<pt.error()  
    <<" \t phi: " << phi.value() <<"+/-"<<phi.error()
    <<" \t tip: " << tip.value() <<"+/-"<<tip.error()
    <<" \t zip: " << zip.value() <<"+/-"<<zip.error()
    <<" \t charge: " << charge;

  float sinTheta = 1/sqrt(1+sqr(cotTheta.value()));
  float cosTheta = cotTheta.value()*sinTheta;
  int tipSign = tip.value() > 0 ? 1 : -1;

  AlgebraicSymMatrix m(5,0);
  float invPtErr = 1./sqr(pt.value()) * pt.error();
  m[0][0] = sqr(sinTheta) * (
              sqr(invPtErr)
            + sqr(cotTheta.error()/pt.value()*cosTheta * sinTheta)
            );
  m[0][2] = sqr( cotTheta.error()) * cosTheta * sqr(sinTheta) / pt.value();
  m[1][1] = sqr( phi.error() );
  m[2][2] = sqr( cotTheta.error());
  m[3][3] = sqr( tip.error() );
  m[4][4] = sqr( zip.error() );
  LocalTrajectoryError error(m);

  LocalTrajectoryParameters lpar(
    LocalPoint(tipSign*tip.value(), -tipSign*zip.value(), 0),
    LocalVector(0., -tipSign*pt.value()*cotTheta.value(), pt.value()),
    charge);

  Surface::RotationType rot(
      sin(phi.value())*tipSign, -cos(phi.value())*tipSign,             0,
                     0,                 0,     -1*tipSign,
      cos(phi.value()),          sin(phi.value()),             0);
  BoundPlane * impPointPlane = new BoundPlane(GlobalPoint(0.,0.,0.), rot);

  TrajectoryStateOnSurface impactPointState( lpar , error, *impPointPlane, mf, 1.0);

  TSCPBuilderNoMaterial tscpBuilder;
  TrajectoryStateClosestToPoint tscp = 
      tscpBuilder(*(impactPointState.freeState()), GlobalPoint(0,0,0) );

  int nhits = hits.size();
  reco::Track * track = new reco::Track( chi2,         // chi2
                          2*nhits-5,  // dof
					 tscp.perigeeParameters(),tscp.pt(),
                   tscp.perigeeError());

  return track;
}
