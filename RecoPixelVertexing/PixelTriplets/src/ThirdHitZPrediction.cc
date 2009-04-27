#include "ThirdHitZPrediction.h"
template<class T> T sqr(T t) { return t * t; }

ThirdHitZPrediction::ThirdHitZPrediction(
    const GlobalPoint& p1, float erroRPhi1, float errorZ1,
    const GlobalPoint& p2, float erroRPhi2, float errorZ2, 
    double radius, double nSigma)
  :
    thePoint1(p1), thePoint2(p2),
    theErrorXY1(erroRPhi1), theErrorZ1(errorZ1),
    theErrorXY2(erroRPhi2), theErrorZ2(errorZ2),
    theRadius(radius), theNSigma(nSigma)
{}

ThirdHitZPrediction::Range ThirdHitZPrediction::operator()(
   const GlobalPoint& thePoint3, float erroRPhi3) const
{
  double dR12 = (thePoint2-thePoint1).perp();
  double dR23 = (thePoint3-thePoint2).perp();
  double dZ12 = thePoint2.z()-thePoint1.z();
  double z3 = thePoint2.z() + dZ12*asin(dR23/2./theRadius)/asin(dR12/2./theRadius);
  double sqr_errorXY12 = sqr(theErrorXY1)+sqr(theErrorXY2);
  double sqr_errorXY23 = sqr(theErrorXY2)+sqr(erroRPhi3);

  double error = sqrt( sqr( (1+dR23/dR12)*theErrorZ2 )
                     + sqr( dR23/dR12 * theErrorZ1 )
                     + sqr(dZ12/dR12 )*sqr_errorXY23 
                     + sqr(dZ12*dR23/dR12/dR12)*sqr_errorXY12
                     );
  error *= theNSigma;
  return Range(z3-error,z3+error);
}

