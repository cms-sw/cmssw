#include "CommonTools/Statistics/src/GammaContinuedFraction.h"
#include <cmath>
#include <iostream>

#define ITMAX 100        // maximum allowed number of iterations
#define EPS 3.0e-7       // relative accuracy
#define FPMIN 1.0e-30    // number near the smallest representable floating-point number

float
GammaContinuedFraction( float a, float x )
{
  int i;
  float an,del;

  /* Set up for evaluating continued fraction by modified Lentz's method (par.5.2
     in Numerical Recipes in C) with b_0 = 0 */
  double b = x+1.0-a;
  double c = 1.0/FPMIN;
  double d = 1.0/b;
  double h = d;
  for (i=1;i<=ITMAX;i++) {
    an = -i*(i-a);
    b += 2.0;
    d=an*d+b;
    if (fabs(d) < FPMIN) d=FPMIN;
    c=b+an/c;
    if (fabs(c) < FPMIN) c=FPMIN;
    d=1.0/d;
    del=d*c;
    h *= del;
    if (fabs(del-1.0) < EPS) break;
  }
  if( i > ITMAX ) std::cerr << "GammaContinuedFraction::a too large, "
		       << "ITMAX too small" << std::endl;
  return h;
}
#undef ITMAX
#undef EPS
#undef FPMIN
/* (C) Copr. 1986-92 Numerical Recipes Software B2.. */
