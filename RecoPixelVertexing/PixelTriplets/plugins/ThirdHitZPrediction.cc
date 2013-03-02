#include "ThirdHitZPrediction.h"

namespace {
  template<class T> T sqr(T t) { return t * t; }
}

ThirdHitZPrediction::ThirdHitZPrediction(
    const GlobalPoint& p1, float erroRPhi1, float errorZ1,
    const GlobalPoint& p2, float erroRPhi2, float errorZ2, 
    double curvature, double nSigma)
  :
    thePoint1(p1), thePoint2(p2),
    theErrorXY1(erroRPhi1), theErrorZ1(errorZ1),
    theErrorXY2(erroRPhi2), theErrorZ2(errorZ2),
    theCurvature(curvature), theNSigma(nSigma)
{}

ThirdHitZPrediction::Range ThirdHitZPrediction::operator()(
   const GlobalPoint& thePoint3, float erroRPhi3) const
{
  double dR23 = (thePoint3-thePoint2).perp();

  double slope = dR23/dR12;
  if (    (theCurvature > 1.e-4) 
	  && (std::abs(0.5*dR23*theCurvature) < 1.) 
	  ) slope = std::asin(0.5*dR23*theCurvature)/dS12;

  double z3 = thePoint2.z() + dZ12*slope;

  double sqr_errorXY23 = sqr_errorXY2+sqr(erroRPhi3);
  double error = sqrt( sqr( (1+dR23/dR12)*theErrorZ2 )
		       + sqr( dR23/dR12 * theErrorZ1 )
		       + sqr(dZ12/dR12 )*sqr_errorXY23 
		       + sqr((dZ12/dR12)*(dR23/dR12))*sqr_errorXY12
		       );
  error *= theNSigma;
  return Range(z3-error,z3+error);
}

