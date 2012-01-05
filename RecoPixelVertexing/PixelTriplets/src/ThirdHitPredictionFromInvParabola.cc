
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvParabola.h"

#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

namespace {
  template <class T> inline T sqr( T t) {return t*t;}
}


typedef Basic2DVector<double> Point2D;
typedef PixelRecoRange<double> Ranged;

using namespace std;

ThirdHitPredictionFromInvParabola::ThirdHitPredictionFromInvParabola( 
    const GlobalPoint& P1, const GlobalPoint& P2,double ip, double curv, double torlerance)
  : theTolerance(torlerance)
{
  init(P1,P2,ip,std::abs(curv));
}


void ThirdHitPredictionFromInvParabola::
    init( const GlobalPoint & P1, const GlobalPoint & P2, double ip, double curv)
{
//  GlobalVector aX = GlobalVector( P2.x()-P1.x(), P2.y()-P1.y(), 0.).unit();
 
  Point2D p1 = P1.basicVector().xy();
  Point2D p2 = P2.basicVector().xy();
  theRotation = Rotation(p1);
  p1 = transform(p1);  // (1./P1.xy().mag(),0); 
  p2 = transform(p2);

 
  u1u2 = p1.x()*p2.x();
  overDu = 1./(p2.x() - p1.x());
  pv = p1.y()*p2.x() - p2.y()*p1.x();
  dv = p2.y() - p1.y();
  su = p2.x() + p1.x();

  RangeD ipRange(-ip, ip); 
  ipRange.sort();
  
  double ipIntyPlus = ipFromCurvature(0.,1);
  double ipCurvPlus = ipFromCurvature(curv, 1);
  double ipCurvMinus = ipFromCurvature(curv, -1);

  
  RangeD ipRangePlus(ipIntyPlus, ipCurvPlus); ipRangePlus.sort();
  RangeD ipRangeMinus(-ipIntyPlus, ipCurvMinus); ipRangeMinus.sort();

  theIpRangePlus  = ipRangePlus.intersection(ipRange);
  theIpRangeMinus = ipRangeMinus.intersection(ipRange);
}
    

ThirdHitPredictionFromInvParabola::Point2D ThirdHitPredictionFromInvParabola::findPointAtCurve(
    double r, int c, double ip) const
{
  //
  // assume u=(1-alpha^2/2)/r v=alpha/r
  // solve qudratic equation neglecting aplha^4 term
  //
  double A = coeffA(ip,c);
  double B = coeffB(ip,c);

  double overR = 1./r;
  double ipOverR = ip*overR;

  double delta = 1-4*(0.5*B+ipOverR)*(-B+A*r-ipOverR);
  double sqrtdelta = (delta > 0) ? std::sqrt(delta) : 0.;
  double alpha = (c>0)?  (-c+sqrtdelta)/(B+2*ipOverR) :  (-c-sqrtdelta)/(B+2*ipOverR);

  double v = alpha*overR;
  double d2 = overR*overR - v*v;
  double u = (d2 > 0) ? std::sqrt(d2) : 0.;

  return Point2D(u,v); // not rotated!
}


ThirdHitPredictionFromInvParabola::Range ThirdHitPredictionFromInvParabola::rangeRPhi(
    double radius, int charge) const
{
  RangeD ip = (charge > 0) ? theIpRangePlus : theIpRangeMinus;

  Point2D pred_tmp1 = findPointAtCurve(radius,charge,ip.min());
  Point2D pred_tmp2 = findPointAtCurve(radius,charge,ip.max());

  double phi1 = theRotation.rotateBack(pred_tmp1).phi();
  double phi2 = phi1+radius*(pred_tmp2.y()-pred_tmp1.y()); 
  
  if (ip.empty()) {
    Range r1(phi1*radius-theTolerance, phi1*radius+theTolerance); 
    Range r2(phi2*radius-theTolerance, phi2*radius+theTolerance); 
    return r1.intersection(r2);
  }

  if (phi2<phi1) std::swap(phi1, phi2); 
  return Range(radius*phi1-theTolerance, radius*phi2+theTolerance);
  
}

/*
ThirdHitPredictionFromInvParabola::Range ThirdHitPredictionFromInvParabola::rangeRPhiSlow(
    double radius, int charge, int nIter) const
{
  Range predRPhi(1.,-1.);

  double invr2 = 1/(radius*radius);
  double u = sqrt(invr2);
  double v = 0.;

  Range ip = (charge > 0) ? theIpRangePlus : theIpRangeMinus;

  for (int i=0; i < nIter; ++i) {
    v = predV(u, ip.min(), charge); 
    double d2 = invr2-sqr(v);
    u = (d2 > 0) ? sqrt(d2) : 0.;
  }
  Point2D  pred_tmp1(u, v);
  double phi1 = transformBack(pred_tmp1).phi(); 
  while ( phi1 >= M_PI) phi1 -= 2*M_PI;
  while ( phi1 < -M_PI) phi1 += 2*M_PI;


  u = sqrt(invr2); 
  v=0;
  for (int i=0; i < nIter; ++i) {
    v = predV(u, ip.max(), charge); 
    double d2 = invr2-sqr(v);
    u = (d2 > 0) ? sqrt(d2) : 0.;
  }
  Point2D  pred_tmp2(u, v);
  double phi2 = transformBack(pred_tmp2).phi(); 
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
*/

