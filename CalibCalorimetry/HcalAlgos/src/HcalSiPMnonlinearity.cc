#include <iostream>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_complex.h>
#include "CalibCalorimetry/HcalAlgos/interface/HcalSiPMnonlinearity.h"

// Assume parameters come to us from the reco side; i.e.,
// true pes = corfun(pixelsfired). But we want to invert that.
//
int HcalSiPMnonlinearity::getPixelsFired(int inpes) const
{
  gsl_complex z[3];
  double w = -inpes;
  // normalize params
  double a = a2/w;
  double b = b1/w;
  double c = c0/w;
  int nroots = gsl_poly_complex_solve_cubic(a, b, c, &z[1], &z[2], &z[3]);
  assert(nroots);

  // all use cases tested over the full range of anticipated values;
  // the first root is always the right one.
  double realpix = 0;
  // find real roots
  for(int i = 0; i < 3; ++i){
    if(z[i].dat[1]==0){
      realpix = z[i].dat[0];
      break;
    }
  }

  return realpix > 0 ? (int)(realpix+0.5) : 0;
}
