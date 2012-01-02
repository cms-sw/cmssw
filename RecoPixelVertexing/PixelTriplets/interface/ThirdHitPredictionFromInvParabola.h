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
  typedef Basic2DVector<double> Point2D;


  ThirdHitPredictionFromInvParabola(const GlobalPoint & P1, const GlobalPoint & P2,
    double ip, double curv, double tolerance);

//  inline Range operator()(double radius, int charge) const { return rangeRPhiSlow(radius,charge,1); } 
  inline Range operator()(double radius, int charge) const { return rangeRPhi(radius,charge); } 

  Range rangeRPhi(double radius, int charge) const;
  // Range rangeRPhiSlow(double radius, int charge, int nIter=5) const;

  void init( const GlobalPoint & P1, const GlobalPoint & P2,  double ip, double curv);
private:

  inline double coeffA(double impactParameter, int charge) const;
  inline double coeffB(double impactParameter, int charge) const;
  inline double predV(double u, double  ip, int charge) const;
  inline double ipFromCurvature(double  curvature, int charge) const;

  Point2D transform(Point2D const & p) const {
    return theRotation.rotate(p)/p.mag2();
  }

  Point2D transformBack(Point2D const & p) const {
    return theRotation.rotateBack(p)/p.mag2();
  }

private:

  Rotation theRotation;
  double u1u2, overDu, pv, dv, su;

  Point2D findPointAtCurve(double radius, int charge, double ip) const dso_internal;

  Range theIpRangePlus, theIpRangeMinus; 
  double theTolerance;

};



double  ThirdHitPredictionFromInvParabola::
    coeffA(double impactParameter, int charge) const
{
  return -charge*pv*overDu - u1u2*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    coeffB(double impactParameter,int charge) const
{
  return charge*dv*overDu - su*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    ipFromCurvature(double curvature, int charge) const 
{
  double overU1u2 = 1./u1u2;
  double inInf = -charge*pv*overDu*overU1u2;
  return inInf-curvature*overU1u2*0.5;
}

double ThirdHitPredictionFromInvParabola::
predV( double u, double ip, int charge) const
{
  return -charge*( coeffA(ip,charge) - coeffB(ip,charge)*u - ip*u*u);
}


#endif
