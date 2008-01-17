#include "ElectroWeakAnalysis/ZMuMu/interface/ConvZShapeGauss.h"

ConvZShapeGauss::ConvZShapeGauss(ZetaShape & zs, Gaussian & gauss, double deltax, int bins):
  zs_(zs), gauss_(gauss) {
  deltax_ = deltax;
  bins_ = bins;
}

double ConvZShapeGauss::operator()(double x) const {
  double dx =  deltax_/bins_;
  double f = 0; double fbw = 0; double gau = 0;
  for( int n = 0; n < bins_; ++n ) {
    //down 
    double y = x + (n+.5) * dx;
    fbw = zs_(y);
    gau = gauss_(x-y);
    f +=  fbw * gau ;
    //up
    double yy = x - (n+.5) * dx ;
    fbw = zs_(yy);
    gau = gauss_(x-yy);
    f +=  fbw * gau ;
  }   
  double BWG = f * deltax_/bins_;
  return BWG;
}
