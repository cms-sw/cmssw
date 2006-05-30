#include "PixelTrackBuilder.h"
#include "DataFormats/TrackReco/interface/HelixParameters.h"
#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Geometry/Surface/interface/BoundPlane.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"


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
/*
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

  
  
   const CartesianTrajectoryError& cte = impactPointState.cartesianError();
   AlgebraicSymMatrix m6 = cte.matrix();
   math::Error<6>::type cov;
   for( int i = 0; i < 6; ++i )
     for( int j = 0; j <= i; ++j )
       cov( i, j ) = m6.fast( i + 1 , j + 1 );


  float valPt = pt.value();
  //
  //momentum
  //
  math::XYZVector mom( valPt*cos( phi.value()),
                       valPt*sin( phi.value()),
                       valPt*cotTheta.value());

  //
  // point of the closest approax to Beam line
  //
//  cout << "TIP value: " <<  tip.value() << endl;
  math::XYZPoint  vtx(  tip.value()*cos( phi.value()),
                        tip.value()*sin( phi.value()),
                        zip.value());

//  cout <<"vertex: " << vtx << endl;
//  cout <<" momentum: " << mom << endl;

  // temporary fix!
  vtx = math::XYZPoint(0.,0.,vtx.z());

  int nhits = hits.size();

  return new reco::Track( chi2,         // chi2
                          2*nhits-5,  // dof
                          nhits,      // foundHits
                          0,
                          0,          //lost hits
                          charge,
                          vtx,
                          mom,
                          cov);
*/
  return 0;

}

