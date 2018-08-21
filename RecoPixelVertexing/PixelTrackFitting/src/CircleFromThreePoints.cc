#include "RecoPixelVertexing/PixelTrackFitting/interface/CircleFromThreePoints.h"

CircleFromThreePoints::CircleFromThreePoints( const GlobalPoint& inner,
                                    const GlobalPoint& mid,
                                    const GlobalPoint& outer,
                                    double precision)
{
  // move to frame where inner.x() == inner.y() == 0;
  Vector2D b( mid.x()-inner.x(), mid.y()-inner.y());
  Vector2D c( outer.x()-inner.x(), outer.y()-inner.y());
  init( b, c, Vector2D( inner.x(), inner.y()), precision);
}

void CircleFromThreePoints::init( const Vector2D& b, const Vector2D& c,
                          const Vector2D& offset, double precision)
{
  double b2 = b.mag2();
  double c2 = c.mag2();

  double oX(0), oY(0);
  bool solved = false;
  if (fabs(b.x()) > fabs(b.y())) {    // solve for y first
    double k = c.x()/b.x();
    double div = 2*(k*b.y() - c.y());
    if (fabs(div) < precision) theCurvature = 0;  // if the 3 points lie on a line
    else {
      oY = (k*b2 - c2) / div;
      oX = b2/(2*b.x()) - b.y()/b.x() * oY;
      solved = true;
    }
  }
  else {    // solve for x first
    double k = c.y()/b.y();
    double div = 2*(k*b.x()-c.x());
    if (fabs(div) < precision) theCurvature = 0;  // if the 3 points lie on a line
    else {
      oX = (k*b2 - c2) / div;
      oY = b2/(2*b.y()) - b.x()/b.y() * oX;
      solved = true;
    }
  }
  if (solved) {
    theCurvature = 1./sqrt(oX*oX + oY*oY);
    double xC = oX + offset.x();
    double yC = oY + offset.y();
    theCenter = Vector2D( xC, yC);
    //    thePhi = acos(xC/sqrt(xC*xC + yC*yC));

    //    if (xC<0.) thePhi = thePhi - PI;
    //    cout << setiosflags(ios::showpoint | ios::fixed);
    //
    //    cout << "CircleFromThreePoints::init curv = " << theCurvature << endl;
    //    cout << "CircleFromThreePoints::init center prime = " << oX << " " << oY << endl;
    //    cout << "CircleFromThreePoints::init offset = " << offset.x() << " " << offset.y() << endl;
    //    cout << "CircleFromThreePoints::init center = " << theCenter.x()<< " " << theCenter.y() << endl;
    //
    //    float d = sqrt(theCenter.x()*theCenter.x()+theCenter.y()*theCenter.y());
    //    cout << "CircleFromThreePoints::init dfloat = " << setw(10) << setprecision(5) << d << endl;
    //    cout << "CircleFromThreePoints::init radius = " << 1/theCurvature << endl;
  }
}

