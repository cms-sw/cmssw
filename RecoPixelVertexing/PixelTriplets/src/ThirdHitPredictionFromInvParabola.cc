
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvParabola.h"

#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

template <class T> T sqr( T t) {return t*t;}

typedef Basic3DVector<double> Point3D;
typedef Basic2DVector<double> Point2D;
typedef PixelRecoRange<double> Ranged;

//ThirdHitPredictionFromInvParabola::
//    ThirdHitPredictionFromInvParabola( const OrderedHitPair & hitPair)
//{
//  GlobalPoint P1 = TrackingRecHit( hitPair.inner()).globalPosition();
//  GlobalPoint P2 = TrackingRecHit( hitPair.outer()).globalPosition();
//  init(P1,P2);
//}

ThirdHitPredictionFromInvParabola::ThirdHitPredictionFromInvParabola( 
    const GlobalPoint& P1, const GlobalPoint& P2,double ip, double curv, double torlerance)
  : theTolerance(torlerance)
{
  init(P1,P2,ip,curv);
}


void ThirdHitPredictionFromInvParabola::
    init( const GlobalPoint & P1, const GlobalPoint & P2, double ip, double curv)
{
  GlobalVector aX = GlobalVector( P2.x()-P1.x(), P2.y()-P1.y(), 0.).unit();
  GlobalVector aY( -aX.y(), aX.x(), 0.); 
  GlobalVector aZ( 0., 0., 1.);
  theRotation = Rotation(aX,aY,aZ); 

  p1 = PointUV(Point2D(P1.x(),P1.y()), &theRotation);
  p2 = PointUV(Point2D(P2.x(),P2.y()), &theRotation);

  ipRange = Range(-ip, ip); 
  ipRange.sort();
  ipConstraint = Range(ipFromCurvature(-curv),ipFromCurvature(curv)); 
  ipConstraint.sort();
  theConstrainedIp = ipRange.intersection(ipConstraint);
}
    
double ThirdHitPredictionFromInvParabola::
    test(const GlobalPoint & P3, const double & ip) const
{
  Point2D point3(P3.x(),P3.y());
  PointUV p3( point3, &theRotation);
  
  double predv = predV(p3.u(), ip);
  PointUV predicted(p3.u(), predv, &theRotation);
  double dphi = predicted.unmap().phi()-point3.phi();
  double dist = (predicted.unmap()-point3).mag();
  //cout <<" ip="<<ip<<" predV="<<predv<<" dphi:"<<dphi<<" dist:"<<dist<<endl;
   
  return dist;
}

ThirdHitPredictionFromInvParabola::Range ThirdHitPredictionFromInvParabola::operator()(
    double radius) const
{
  double invr2 = 1/radius/radius; 
  double u = 1./radius;
  double v = 0.;
  
  int nIter=2;                               // here even 1 works well
  for (int i=0; i<nIter; i++) {
    v = predV(u,theConstrainedIp.mean()); 
    double d2 = invr2-sqr(v);
    u = (d2 > 0) ? sqrt(d2) : 0.;
  }
  return rangeRPhi(radius,u);
}


ThirdHitPredictionFromInvParabola::Range ThirdHitPredictionFromInvParabola::operator()(
    const GlobalPoint& hit) const
{
  Point2D point3(hit.x(),hit.y());
  PointUV p3( point3, &theRotation);
  return rangeRPhi( point3.mag(), p3.u());
}

ThirdHitPredictionFromInvParabola::Range ThirdHitPredictionFromInvParabola::rangeRPhi(double radius, double pointUV_u) const
{
  Range predv;
  if (!theConstrainedIp.empty()) {
    double tmp1 = predV(pointUV_u, theConstrainedIp.min());
    double tmp2 = predV(pointUV_u, theConstrainedIp.max());
    predv = Range(tmp1,tmp2);
    predv.sort();
  }
  else if (ipRange.max() < ipConstraint.min()) {
    double tmp = predV(pointUV_u, ipRange.max());
    predv = Range(tmp,tmp);
  }
  else {
    double tmp = predV(pointUV_u, ipRange.min());
    predv = Range(tmp,tmp);
  }

  PointUV predicted(pointUV_u, predv.min(), &theRotation);
  double phi1 = predicted.unmap().phi();
  if ( phi1 >= M_PI) phi1 -= 2*M_PI;
  if ( phi1 < -M_PI) phi1 += 2*M_PI;
  double phi2 = phi1+radius*(predv.max()-predv.min());
  return Range( radius*phi1-theTolerance, radius*phi2+theTolerance);
} 


double ThirdHitPredictionFromInvParabola::
    ipFromCurvature(const double & curvature) const 
{
  double u1u2 = p1.u()*p2.u();
  double du = p2.u() - p1.u();
  double pv = p1.v()*p2.u() - p2.v()*p1.u();
  return (pv/du + curvature/2.)/u1u2;
}

double ThirdHitPredictionFromInvParabola::
    coeffA(const double & impactParameter) const 
{
  double u1u2 = p1.u()*p2.u();
  double du = p2.u() - p1.u();
  double pv = p1.v()*p2.u() - p2.v()*p1.u();
  return pv/du - u1u2*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    coeffB(const double & impactParameter) const 
{
  double dv = p2.v() - p1.v();
  double du = p2.u() - p1.u(); 
  double su = p2.u() + p1.u();
  return -dv/du - su*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    predV( const double & u, const double & ip) const 
{
  return coeffA(ip) - coeffB(ip)*u - ip*sqr(u); 
}
