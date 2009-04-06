#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvLine.h"

#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

template <class T> T sqr( T t) {return t*t;}

typedef Basic3DVector<double> Point3D;
typedef Basic2DVector<double> Point2D;
typedef PixelRecoRange<double> Ranged;

using namespace std;

ThirdHitPredictionFromInvLine::ThirdHitPredictionFromInvLine(
    const GlobalPoint & P1, const GlobalPoint & P2)
{
  GlobalVector aX = GlobalVector( P1.x(), P1.y(), 0.).unit();
  GlobalVector aY( -aX.y(), aX.x(), 0.);
  GlobalVector aZ( 0., 0., 1.);
  theRotation = Rotation(aX,aY,aZ);

  p1 = PointUV(Point2D(P1.x(),P1.y()), &theRotation);
  p2 = PointUV(Point2D(P2.x(),P2.y()), &theRotation);

}

GlobalPoint ThirdHitPredictionFromInvLine::center() const
{
  double den=2.*(p1.v()*p2.u()-p1.u()*p2.v());
  Point3D tmp = theRotation.multiplyInverse( Point2D( (p1.v()-p2.v())/den, (p2.u()-p1.u())/den ) );
  return GlobalPoint(tmp.x(), tmp.y(), 0.);
}
