#ifndef ThirdHitPredictionFromInvParabola_H
#define ThirdHitPredictionFromInvParabola_H

/** Check phi compatibility of a hit with a track
    constrained by a hit pair and TrackingRegion kinematical constraints.
    The "inverse parabola method" is used. 
    M.Hansroul, H.Jeremie, D.Savard NIM A270 (1998) 490. 
 */
    
  

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "FWCore/Utilities/interface/Visibility.h"


class TrackingRegion;
class OrderedHitPair;


class ThirdHitPredictionFromInvParabola {

public:

  typedef TkRotation2D<double> Rotation;
  typedef PixelRecoRange<float> Range;
  typedef PixelRecoRange<double> RangeD;
  typedef Basic2DVector<double> Point2D;


  ThirdHitPredictionFromInvParabola(const GlobalPoint & P1, const GlobalPoint & P2,
    double ip, double curv, double tolerance);

//  inline Range operator()(double radius, int charge) const { return rangeRPhiSlow(radius,charge,1); } 
  inline Range operator()(double radius, int charge) const { return rangeRPhi(radius,charge); } 

  Range rangeRPhi(double radius, int charge) const __attribute__ ((optimize(3, "fast-math")));
  // Range rangeRPhiSlow(double radius, int charge, int nIter=5) const;

  void init( const GlobalPoint & P1, const GlobalPoint & P2,  double ip, double curv);
private:

  inline double coeffA(double impactParameter, double charge) const;
  inline double coeffB(double impactParameter, double charge) const;
  inline double predV(double u, double  ip, double charge) const;
  inline double ipFromCurvature(double  curvature, double charge) const;

  Point2D transform(Point2D const & p) const {
    return theRotation.rotate(p)/p.mag2();
  }

  Point2D transformBack(Point2D const & p) const {
    return theRotation.rotateBack(p)/p.mag2();
  }

private:

  Rotation theRotation;
  double u1u2, overDu, pv, dv, su;

  inline void findPointAtCurve(double radius, double charge, double ip, double &u, double &v) const;

  RangeD theIpRangePlus, theIpRangeMinus; 
  double theTolerance;

};



double  ThirdHitPredictionFromInvParabola::
    coeffA(double impactParameter, double charge) const
{
  return -charge*pv*overDu - u1u2*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    coeffB(double impactParameter, double charge) const
{
  return charge*dv*overDu - su*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    ipFromCurvature(double curvature, double charge) const 
{
  double overU1u2 = 1./u1u2;
  double inInf = -charge*pv*overDu*overU1u2;
  return inInf-curvature*overU1u2*0.5;
}

double ThirdHitPredictionFromInvParabola::
predV( double u, double ip, double charge) const
{
  return -charge*( coeffA(ip,charge) - coeffB(ip,charge)*u - ip*u*u);
}

void ThirdHitPredictionFromInvParabola::findPointAtCurve(double r, double c, double ip, 
							 double & u, double & v) const
{
  //
  // assume u=(1-alpha^2/2)/r v=alpha/r
  // solve qudratic equation neglecting aplha^4 term
  //
  double A = coeffA(ip,c);
  double B = coeffB(ip,c);

  // double overR = 1./r;
  double ipOverR = ip/r; // *overR;

  double delta = 1-4*(0.5*B+ipOverR)*(-B+A*r-ipOverR);
  // double sqrtdelta = (delta > 0) ? std::sqrt(delta) : 0.;
  double sqrtdelta = std::sqrt(delta);
  double alpha = (c>0)?  (-c+sqrtdelta)/(B+2*ipOverR) :  (-c-sqrtdelta)/(B+2*ipOverR);

  v = alpha;  // *overR
  double d2 = 1. - v*v;  // overR*overR - v*v
  // u = (d2 > 0) ? std::sqrt(d2) : 0.;
  u = std::sqrt(d2);

  // u,v not rotated! not multiplied by 1/r
}


#endif
