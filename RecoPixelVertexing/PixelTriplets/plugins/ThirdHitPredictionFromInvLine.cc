#include "ThirdHitPredictionFromInvLine.h"

#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

template <class T>
T sqr(T t) {
  return t * t;
}

typedef Basic3DVector<double> Point3D;
typedef Basic2DVector<double> Point2D;
typedef PixelRecoRange<double> Ranged;

#include <iostream>

using namespace std;

ThirdHitPredictionFromInvLine::ThirdHitPredictionFromInvLine(const GlobalPoint &P1,
                                                             const GlobalPoint &P2,
                                                             double errorRPhiP1,
                                                             double errorRPhiP2)
    : nPoints(0),
      theSum(0.),
      theSumU(0.),
      theSumUU(0.),
      theSumV(0.),
      theSumUV(0.),
      theSumVV(0.),
      hasParameters(false),
      theCurvatureValue(0.),
      theCurvatureError(0.),
      theChi2(0.) {
  GlobalVector aX = GlobalVector(P1.x(), P1.y(), 0.).unit();
  GlobalVector aY(-aX.y(), aX.x(), 0.);
  GlobalVector aZ(0., 0., 1.);
  theRotation = Rotation(aX, aY, aZ);

  add(P1, errorRPhiP1);
  add(P2, errorRPhiP2);
}

GlobalPoint ThirdHitPredictionFromInvLine::crossing(double radius) const {
  double A = -(theSum * theSumUV - theSumU * theSumV) / (sqr(theSumU) - theSum * theSumUU);
  double B = (theSumU * theSumUV - theSumUU * theSumV) / (sqr(theSumU) - theSum * theSumUU);
  double delta = sqr(2. * A * B) - 4 * (1 + sqr(A)) * (sqr(B) - sqr(1 / radius));
  double sqrtdelta = (delta > 0.) ? sqrt(delta) : 0.;
  double u1 = (-2. * A * B + sqrtdelta) / 2. / (1 + sqr(A));
  double v1 = A * u1 + B;
  Point2D tmp = PointUV(u1, v1, &theRotation).unmap();
  return GlobalPoint(tmp.x(), tmp.y(), 0.);
}

void ThirdHitPredictionFromInvLine::add(const GlobalPoint &p, double errorRPhi) {
  double weigth = sqr(sqr(p.perp()) / errorRPhi);
  add(PointUV(Point2D(p.x(), p.y()), &theRotation), weigth);
}

void ThirdHitPredictionFromInvLine::add(const ThirdHitPredictionFromInvLine::PointUV &point, double weigth) {
  hasParameters = false;
  nPoints++;
  theSum += weigth;
  theSumU += point.u() * weigth;
  theSumUU += sqr(point.u()) * weigth;
  theSumV += point.v() * weigth;
  theSumUV += point.u() * point.v() * weigth;
  theSumVV += sqr(point.v()) * weigth;
  check();
}

void ThirdHitPredictionFromInvLine::remove(const GlobalPoint &p, double errorRPhi) {
  hasParameters = false;
  PointUV point(Point2D(p.x(), p.y()), &theRotation);
  double weigth = sqr(sqr(p.perp()) / errorRPhi);
  nPoints--;
  theSum -= weigth;
  theSumU -= point.u() * weigth;
  theSumUU -= sqr(point.u()) * weigth;
  theSumV -= point.v() * weigth;
  theSumUV -= point.u() * point.v() * weigth;
  theSumVV -= sqr(point.v()) * weigth;
  check();
}

void ThirdHitPredictionFromInvLine::print() const {
  std::cout << " nPoints: " << nPoints << " theSumU: " << theSumU << " theSumUU: " << theSumUU
            << " theSumV: " << theSumV << " theSumUV: " << theSumUV << std::endl;
}

void ThirdHitPredictionFromInvLine::check() {
  if (hasParameters)
    return;

  long double D = theSumUU * theSum - theSumU * theSumU;
  long double A = (theSumUV * theSum - theSumU * theSumV) / D;
  long double B = (theSumUU * theSumV - theSumUV * theSumU) / D;
  double rho = 2. * fabs(B) / sqrt(1 + sqr(A));
  double sigmaA2 = theSum / D;
  double sigmaB2 = theSumUU / D;

  hasParameters = true;
  theCurvatureError = sqrt(sqr(rho / B) * sigmaB2 + sqr(rho / (1 + sqr(A))) * sigmaA2);
  theCurvatureValue = 2. * fabs(B) / sqrt(1 + sqr(A));
  theChi2 = theSumVV - 2 * A * theSumUV - 2 * B * theSumV + 2 * A * B * theSumU + B * B * theSum + A * A * theSumUU;
}

/*
GlobalPoint ThirdHitPredictionFromInvLine::center() const
{
  long double den = theSumU*theSumUV - theSumUU*theSumV; 
  double a = (theSum*theSumUV-theSumU*theSumV)/2./den;
  double b = (sqr(theSumU)-theSum*theSumUU)/2./den;
  Point3D tmp = theRotation.multiplyInverse( Point2D(a,b) );
  return GlobalPoint(tmp.x(), tmp.y(), 0.);
}
*/
