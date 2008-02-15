#include "PhysicsTools/Utilities/interface/ConvGaussZLineShape.h"
using namespace function;

double ConvGaussZLineShape::operator()(double x) const {
  double deltax = 6* (*sigma); //from -3*sigma to +3*sigma
  int bins = 100; //enough...
  double dx = deltax/bins;
  double f = 0, fbw, gau, y;
  for(int n = 0; n < bins; ++n) {
    //down 
    y = x + (n+.5) * dx;
    fbw = zs_(y);
    gau = gauss_(x - y);
    f +=  fbw * gau;
    //up
    y = x - (n+.5) * dx;
    fbw = zs_(y);
    gau = gauss_(x - y);
    f +=  fbw * gau;
  }   
  return f * deltax/bins;
}

