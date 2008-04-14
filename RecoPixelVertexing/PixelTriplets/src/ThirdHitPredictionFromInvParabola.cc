
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

using namespace std;

ThirdHitPredictionFromInvParabola::ThirdHitPredictionFromInvParabola( 
    const GlobalPoint& P1, const GlobalPoint& P2,double ip, double curv, double torlerance)
  : theTolerance(torlerance)
{
  init(P1,P2,ip,fabs(curv));
}


void ThirdHitPredictionFromInvParabola::
    init( const GlobalPoint & P1, const GlobalPoint & P2, double ip, double curv)
{
//  GlobalVector aX = GlobalVector( P2.x()-P1.x(), P2.y()-P1.y(), 0.).unit();
  GlobalVector aX = GlobalVector( P1.x(), P1.y(), 0.).unit();
  GlobalVector aY( -aX.y(), aX.x(), 0.); 
  GlobalVector aZ( 0., 0., 1.);
  theRotation = Rotation(aX,aY,aZ); 

  p1 = PointUV(Point2D(P1.x(),P1.y()), &theRotation);
  p2 = PointUV(Point2D(P2.x(),P2.y()), &theRotation);

  Range ipRange(-ip, ip); 
  ipRange.sort();
  
  double ipIntyPlus = ipFromCurvature(0.,1);
  double ipCurvPlus = ipFromCurvature(fabs(curv), 1);
  double ipCurvMinus = ipFromCurvature(fabs(curv), -1);

  
  Range ipRangePlus = Range(ipIntyPlus, ipCurvPlus); ipRangePlus.sort();
  Range ipRangeMinus = Range(-ipIntyPlus, ipCurvMinus); ipRangeMinus.sort();

  theIpRangePlus  = ipRangePlus.intersection(ipRange);
  theIpRangeMinus = ipRangeMinus.intersection(ipRange);
}
    
ThirdHitPredictionFromInvParabola::Range ThirdHitPredictionFromInvParabola::operator()(
    double radius, int charge) const
{
  Range predRPhi(1.,-1.);

  double invr2 = 1/radius/radius;
  double u = invr2;
  double v = 0.;
  int nIter=10;

  Range ip = (charge > 0) ? theIpRangePlus : theIpRangeMinus;

  for (int i=0; i < nIter; ++i) {
    v = predV(u, ip.min(), charge); 
    double d2 = invr2-sqr(v);
    u = (d2 > 0) ? sqrt(d2) : 0.;
  }
  PointUV  pred_tmp1(u, v,  &theRotation);
  double phi1 = pred_tmp1.unmap().phi(); 
  while ( phi1 >= M_PI) phi1 -= 2*M_PI;
  while ( phi1 < -M_PI) phi1 += 2*M_PI;


  for (int i=0; i < nIter; ++i) {
    v = predV(u, ip.max(), charge); 
    double d2 = invr2-sqr(v);
    u = (d2 > 0) ? sqrt(d2) : 0.;
  }
  PointUV  pred_tmp2(u, v,  &theRotation);
  double phi2 = pred_tmp2.unmap().phi(); 
  while ( phi2-phi1 >= M_PI) phi2 -= 2*M_PI;
  while ( phi2-phi1 < -M_PI) phi2 += 2*M_PI;

// check faster alternative, without while(..) it is enough to:
//  phi2 = phi1+radius*(pred_tmp2.v()-pred_tmp1.v()); 

  if (ip.empty()) {
    Range r1(phi1*radius-theTolerance, phi1*radius+theTolerance); 
    Range r2(phi2*radius-theTolerance, phi2*radius+theTolerance); 
    predRPhi = r1.intersection(r2);
  } else {
    Range r(phi1, phi2); 
    r.sort();
    predRPhi= Range(radius*r.min()-theTolerance, radius*r.max()+theTolerance);
  }
  return predRPhi;

}

double ThirdHitPredictionFromInvParabola::
    ipFromCurvature(const double & curvature, int charge) const 
{
  double u1u2 = p1.u()*p2.u();
  double du = p2.u() - p1.u();
  double pv = p1.v()*p2.u() - p2.v()*p1.u();

  double inInf = -charge*pv/du/u1u2;
  return inInf-curvature/2./u1u2;
}

double  ThirdHitPredictionFromInvParabola::
    coeffA(const double & impactParameter, int charge) const
{
  double u1u2 = p1.u()*p2.u();
  double du = p2.u() - p1.u();
  double pv = p1.v()*p2.u() - p2.v()*p1.u();
  return -charge*pv/du - u1u2*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    coeffB(const double & impactParameter,int charge) const
{
  double dv = p2.v() - p1.v();
  double du = p2.u() - p1.u();
  double su = p2.u() + p1.u();
  return charge*dv/du - su*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    predV( const double & u, const double & ip, int charge) const
{
  return -charge*( coeffA(ip,charge) - coeffB(ip,charge)*u - ip*sqr(u));
}
