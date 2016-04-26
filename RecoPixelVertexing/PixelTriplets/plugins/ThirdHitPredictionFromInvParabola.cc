
#include "ThirdHitPredictionFromInvParabola.h"

#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include "DataFormats/Math/interface/approx_atan2.h"
namespace {
  inline
  float f_atan2f(float y, float x) { return unsafe_atan2f<7>(y,x); }
  template<typename V> inline float f_phi(V v) { return f_atan2f(v.y(),v.x());}
}

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
  Scalar phi1 = f_phi(theRotation.rotateBack(Point2D(u[0],v[0])));
  Scalar phi2 = phi1+(v[1]-v[0]); 
  
  if (phi2<phi1) std::swap(phi1, phi2);

  if (ip.empty()) {
    Range r1(phi1*radius-theTolerance, phi1*radius+theTolerance); 
    Range r2(phi2*radius-theTolerance, phi2*radius+theTolerance); 
    return r1.intersection(r2); // this range can be empty
  }

  return Range(radius*phi1-theTolerance, radius*phi2+theTolerance);
  
}


ThirdHitPredictionFromInvParabola::Range
ThirdHitPredictionFromInvParabola::rangeRPhi(Scalar radius) const
{

  auto getRange = [&](Scalar phi1, Scalar phi2, bool empty)->RangeD {
  
    if (phi2<phi1) std::swap(phi1, phi2);  
    if (empty) {
      RangeD r1(phi1*radius-theTolerance, phi1*radius+theTolerance); 
      RangeD r2(phi2*radius-theTolerance, phi2*radius+theTolerance); 
      return r1.intersection(r2);
    }
    
    return RangeD(radius*phi1-theTolerance, radius*phi2+theTolerance);
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
    phi1[i] = f_atan2f(y,x);
    phi2[i] = phi1[i]+(v[i+2]-v[i]); 
  }

  return getRange(phi1[1],phi2[1],emptyM).sum(getRange(phi1[0],phi2[0],emptyP));
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

