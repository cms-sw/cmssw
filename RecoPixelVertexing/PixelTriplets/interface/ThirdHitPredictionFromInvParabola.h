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


class TrackingRegion;
class OrderedHitPair;


class ThirdHitPredictionFromInvParabola {

public:

  typedef TkRotation<double> Rotation;
  typedef PixelRecoRange<float> Range;

  ThirdHitPredictionFromInvParabola(const GlobalPoint & P1, const GlobalPoint & P2,
    double ip, double curv, double tolerance);

//  inline Range operator()(double radius, int charge) const { return rangeRPhiSlow(radius,charge,1); } 
  inline Range operator()(double radius, int charge) const { return rangeRPhi(radius,charge); } 

  Range rangeRPhi(double radius, int charge) const;
  Range rangeRPhiSlow(double radius, int charge, int nIter=5) const;

  void init( const GlobalPoint & P1, const GlobalPoint & P2,  double ip, double curv);
private:

  inline double coeffA(double impactParameter, int charge) const;
  inline double coeffB(double impactParameter, int charge) const;
  double predV(double u, double  ip, int charge) const;
  inline double ipFromCurvature(double  curvature, int charge) const;


private:

  template <class T> class MappedPoint {
  public:
    MappedPoint() : theU(0), theV(0), pRot(0) { }
    MappedPoint(const T & aU, const T & aV, const TkRotation<T> * aRot) 
        : theU(aU), theV(aV), pRot(aRot) { }
    MappedPoint(const Basic2DVector<T> & point, const TkRotation<T> * aRot)
        : pRot(aRot) {
      T invRadius2 = T(1)/point.mag2();
      Basic3DVector<T> rotated = (*pRot) * point;
      theU = rotated.x() * invRadius2;
      theV = rotated.y() * invRadius2;
    }
    T u() const {return theU; } 
    T v() const {return theV; }
    Basic2DVector<T> unmap () const {
      T radius2 = T(1)/(theU*theU+theV*theV); 
       Basic3DVector<T> tmp
           = (*pRot).multiplyInverse(Basic2DVector<T>(theU,theV));
       return Basic2DVector<T>( tmp.x()*radius2, tmp.y()*radius2);
    }
  private:
    T theU, theV;
    const TkRotation<T> * pRot;
  };

private:

  Rotation theRotation;
  typedef MappedPoint<double> PointUV;
  //PointUV p1, p2;
  double u1u2, overDu, pv, dv, su;

  PointUV findPointAtCurve(double radius, int charge, double ip) const;

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

#endif
