#include "PhysicsTools/Utilities/interface/ConvGaussZLineShape.h"

ConvGaussZLineShape::ConvGaussZLineShape(double m, double g, double Nf, double Ni,
					 double mean, double sigma, double deltax, int bins) :
  zs_(m, g, Nf, Ni), 
  gauss_(mean, sigma),
  deltax_(deltax),
  bins_(bins),
  dx_(deltax_/bins_) {
}

void ConvGaussZLineShape::setParameters(double m, double g, double Nf, double Ni,
					double mean, double sigma, double deltax, int bins) {
  deltax_ = deltax;
  bins_ = bins;
  dx_ = deltax_/bins_;
  zs_.setParameters(m, g, Nf, Ni);
  gauss_.setParameters(mean, sigma); 
}

double ConvGaussZLineShape::operator()(double x) const {
  double f = 0, fbw, gau, y;
  for(int n = 0; n < bins_; ++n) {
    //down 
    y = x + (n+.5) * dx_;
    fbw = zs_(y);
    gau = gauss_(x - y);
    f +=  fbw * gau;
    //up
    y = x - (n+.5) * dx_;
    fbw = zs_(y);
    gau = gauss_(x - y);
    f +=  fbw * gau;
  }   
  return f * deltax_/bins_;
}
