
#include "ThirdHitPredictionFromInvParabola.h"

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

using namespace std;

ThirdHitPredictionFromInvParabola::ThirdHitPredictionFromInvParabola( 
    const GlobalPoint& P1, const GlobalPoint& P2,Scalar ip, Scalar curv, Scalar tolerance)
  : theTolerance(tolerance)
{
  init(P1,P2,ip,std::abs(curv));
}


void ThirdHitPredictionFromInvParabola:: init(Scalar x1,Scalar y1, Scalar x2,Scalar y2,  Scalar ip, Scalar curv) {
//  GlobalVector aX = GlobalVector( P2.x()-P1.x(), P2.y()-P1.y(), 0.).unit();
 
  Point2D p1(x1,y1);
  Point2D p2(x2,y2);
  theRotation = Rotation(p1);
  p1 = transform(p1);  // (1./P1.xy().mag(),0); 
  p2 = transform(p2);

 
  u1u2 = p1.x()*p2.x();
  overDu = 1./(p2.x() - p1.x());
  pv = p1.y()*p2.x() - p2.y()*p1.x();
  dv = p2.y() - p1.y();
  su = p2.x() + p1.x();

  ip = std::abs(ip);
  RangeD ipRange(-ip, ip); 

  
  Scalar ipIntyPlus = ipFromCurvature(0.,true);
  Scalar ipCurvPlus = ipFromCurvature(curv, true);
  Scalar ipCurvMinus = ipFromCurvature(curv, false);

  
  RangeD ipRangePlus(std::min(ipIntyPlus, ipCurvPlus),std::max(ipIntyPlus, ipCurvPlus)); 
  RangeD ipRangeMinus(std::min(-ipIntyPlus, ipCurvMinus),std::max(-ipIntyPlus, ipCurvMinus));

  theIpRangePlus  = ipRangePlus.intersection(ipRange);
  theIpRangeMinus = ipRangeMinus.intersection(ipRange);

  emptyP =  theIpRangePlus.empty();
  emptyM =  theIpRangeMinus.empty();

}
    


ThirdHitPredictionFromInvParabola::Range 
ThirdHitPredictionFromInvParabola::rangeRPhi(Scalar radius, int icharge) const
{
  bool pos =  icharge>0;

  RangeD ip = (pos) ? theIpRangePlus : theIpRangeMinus;


  //  it will vectorize with gcc 4.7 (with -O3 -fno-math-errno)
  // change sign as intersect assume -ip for negative charge...
  Scalar ipv[2]={(pos)? ip.min() : -ip.min() ,(pos)? ip.max() : -ip.max()};
  Scalar u[2], v[2];
  for (int i=0; i!=2; ++i)
    findPointAtCurve(radius,ipv[i],u[i],v[i]);

  // 
  Scalar phi1 = theRotation.rotateBack(Point2D(u[0],v[0])).barePhi();
  Scalar phi2 = phi1+(v[1]-v[0]); 
  
  if (ip.empty()) {
    Range r1(phi1*radius-theTolerance, phi1*radius+theTolerance); 
    Range r2(phi2*radius-theTolerance, phi2*radius+theTolerance); 
    return r1.intersection(r2);
  }

  if (phi2<phi1) std::swap(phi1, phi2); 
  return Range(radius*phi1-theTolerance, radius*phi2+theTolerance);
  

}


namespace {

  // original from chephes single precision version
  template<typename T>
  inline T atan2_fast( T y, T x ) {
  
   constexpr T PIO4F = 0.7853981633974483096;
   constexpr T PIF = 3.141592653589793238;
   constexpr T PIO2F = 1.5707963267948966192;

    // move in first octant
    T xx = std::abs(x);
    T yy = std::abs(y);
    T tmp = T(0);
    if (yy>xx) {
      tmp = yy;
      yy=xx; xx=tmp;
    }
    T t=yy/xx;
    T z = 
      ( t > T(0.4142135623730950) ) ?  // * tan pi/8 
      (t-T(1.0))/(t+T(1.0)) : t;
     

    //printf("%e %e %e %e\n",yy,xx,t,z);
    T z2 = z * z;
    T ret = // (y==0) ? 0 :  // no protection for (0,0)
      (((  T(8.05374449538e-2) * z2
	   - T(1.38776856032E-1)) * z2
	+ T(1.99777106478E-1)) * z2
       - T(3.33329491539E-1)) * z2 * z
      + z;

    // move back in place
    if( t > T(0.4142135623730950) ) ret += PIO4F;
    if (tmp!=0) ret = PIO2F - ret;
    if (x<0) ret = PIF - ret;
    if (y<0) ret = -ret;
    
    return ret;

  }




}


ThirdHitPredictionFromInvParabola::Range
ThirdHitPredictionFromInvParabola::rangeRPhi(Scalar radius) const
{

  auto getRange = [&](Scalar phi1, Scalar phi2, bool empty)->RangeD {
    
    if (empty) {
      RangeD r1(phi1*radius-theTolerance, phi1*radius+theTolerance); 
      RangeD r2(phi2*radius-theTolerance, phi2*radius+theTolerance); 
      return r1.intersection(r2);
    }
    
    return RangeD(radius*std::min(phi1,phi2)-theTolerance, radius*std::max(phi1,phi2)+theTolerance);
  };


  //  it will vectorize with gcc 4.7 (with -O3 -fno-math-errno)
  // change sign as intersect assume -ip for negative charge...
  Scalar ipv[4]={theIpRangePlus.min(), -theIpRangeMinus.min(), theIpRangePlus.max(),  -theIpRangeMinus.max()};
  Scalar u[4], v[4];
  for (int i=0; i<4; ++i)
    findPointAtCurve(radius,ipv[i],u[i],v[i]);

  // 
  auto xr = theRotation.x();
  auto yr = theRotation.y();

  Scalar phi1[2],phi2[2];
  for (int i=0; i<2; ++i) {
    auto x =  xr[0]*u[i] + yr[0]*v[i];
    auto y =  xr[1]*u[i] + yr[1]*v[i];
    phi1[i] = atan2_fast(y,x);
    phi2[i] = phi1[i]+(v[i+2]-v[i]); 
  }


  return getRange(phi1[1],phi2[1],emptyM).sum(getRange(phi1[0],phi2[0],emptyP));

  /*
  Scalar phi1P = theRotation.rotateBack(Point2D(u[0],v[0])).barePhi();
  Scalar phi2P= phi1P+(v[2]-v[0]); 

  Scalar phi1M = theRotation.rotateBack(Point2D(u[1],v[1])).barePhi();
  Scalar phi2M = phi1M+(v[3]-v[1]); 


  return getRange(phi1M,phi2M,emptyM).sum(getRange(phi1P,phi2P,emptyP));

  */

}


/*
ThirdHitPredictionFromInvParabola::Range ThirdHitPredictionFromInvParabola::rangeRPhiSlow(
    Scalar radius, int charge, int nIter) const
{
  Range predRPhi(1.,-1.);

  Scalar invr2 = 1/(radius*radius);
  Scalar u = sqrt(invr2);
  Scalar v = 0.;

  Range ip = (charge > 0) ? theIpRangePlus : theIpRangeMinus;

  for (int i=0; i < nIter; ++i) {
    v = predV(u, charge*ip.min()); 
    Scalar d2 = invr2-sqr(v);
    u = (d2 > 0) ? sqrt(d2) : 0.;
  }
  Point2D  pred_tmp1(u, v);
  Scalar phi1 = transformBack(pred_tmp1).phi(); 
  while ( phi1 >= M_PI) phi1 -= 2*M_PI;
  while ( phi1 < -M_PI) phi1 += 2*M_PI;


  u = sqrt(invr2); 
  v=0;
  for (int i=0; i < nIter; ++i) {
    v = predV(u, ip.max(), charge); 
    Scalar d2 = invr2-sqr(v);
    u = (d2 > 0) ? sqrt(d2) : 0.;
  }
  Point2D  pred_tmp2(u, v);
  Scalar phi2 = transformBack(pred_tmp2).phi(); 
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

