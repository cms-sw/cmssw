
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvParabola.h"

#include <cmath>
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

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
    const GlobalPoint& P1, const GlobalPoint& P2)
{
  init(P1,P2);
}


void ThirdHitPredictionFromInvParabola::
    init( const GlobalPoint & P1, const GlobalPoint & P2)
{
  GlobalVector aX = GlobalVector( P2.x()-P1.x(), P2.y()-P1.y(), 0.).unit();
  GlobalVector aY( -aX.y(), aX.x(), 0.); 
  GlobalVector aZ( 0., 0., 1.);
  theRotation = Rotation(aX,aY,aZ); 

  p1 = PointUV(Point2D(P1.x(),P1.y()), &theRotation);
  p2 = PointUV(Point2D(P2.x(),P2.y()), &theRotation);

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
  cout <<" ip="<<ip<<" predV="<<predv<<" dphi:"<<dphi<<" dist:"<<dist<<endl;
   
  return dist;
}

double ThirdHitPredictionFromInvParabola::
//    isCompatible(const GlobalPoint& hit, const TrackingRegion & region) const
  isCompatible(const GlobalPoint& hit, double ip, double curv) const
{
//  GlobalPoint P3 = hit.globalPosition();
  GlobalPoint P3 = hit;
  Point2D point3(P3.x(),P3.y());
  PointUV p3( point3, &theRotation);

//  double ip = region.originRBound();
//  double curv = PixelRecoUtilities::curvature(1/region.ptMin());
  
  Ranged ipRange(-ip, ip); 
  ipRange.sort();
  Ranged ipConstraint(ipFromCurvature(-curv),ipFromCurvature(curv));
  ipConstraint.sort();
  Ranged ipConstrainedRange = ipRange.intersection(ipConstraint);

  Ranged predv;
  if (!ipConstrainedRange.empty()) {
    double tmp1 = predV(p3.u(), ipConstrainedRange.min());
    double tmp2 = predV(p3.u(), ipConstrainedRange.max()); 
    predv = Ranged(tmp1,tmp2); 
    predv.sort();
  } 
  else if (ipRange.max() < ipConstraint.min()) {
    double tmp = predV(p3.u(), ipRange.max());
    predv = Ranged(tmp,tmp);
  } 
  else {
    double tmp = predV(p3.u(), ipRange.min());
    predv = Ranged(tmp,tmp);
  }

  double distance = 0;
  if ( p3.v() < predv.min()) {
    PointUV predicted(p3.u(), predv.min(), &theRotation);
    double dphi = predicted.unmap().phi()-point3.phi();
    while ( dphi >= M_PI) dphi -= 2*M_PI; 
    while ( dphi < -M_PI) dphi += 2*M_PI; 
    distance = dphi * point3.mag();
  } else if ( predv.max() < p3.v()) {
    PointUV predicted(p3.u(), predv.max(), &theRotation);
    double dphi = predicted.unmap().phi()-point3.phi();
    while ( dphi >= M_PI) dphi -= 2*M_PI; 
    while ( dphi < -M_PI) dphi += 2*M_PI; 
    distance = dphi * point3.mag();
  } 

//
//cout << "ipRange     : "<<ipRange<<endl;
//cout << "ipConstraint: "<<ipConstraint<<endl;
//cout << "ipConstrainedRange: "<<ipConstrainedRange<<endl;
//cout << "predv Range: "<<predv<<" value:"<<p3.v()
//     << " distance: "<< distance << endl;
//

  return distance;
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
