#include "CommonTools/Statistics/src/GammaSeries.h"
#include <iostream>
#include <cmath>

#define ITMAX 100             // maximum allowed number of iterations
#define EPS 3.0e-7            // relative accuracy

float GammaSeries( float a, float x )
{
  if( x < 0.0 ) 
    std::cerr << "GammaSeries::negative argument x" << std::endl;

  if( x == 0. )
    return 0.;

  if( a == 0. ) // this happens at the end, but save all the iterations
    return 0.;

  // coefficient c_n of x^n is Gamma(a)/Gamma(a+1+n), which leads to the
  // recurrence relation c_n = c_(n-1) / (a+n-1) with c_0 = 1/a
  double term = 1/a;
  double sum = term;
  double aplus = a;
  for( int index = 1; index <= ITMAX; index++) {
    ++aplus;
    term *= x/aplus;
    sum += term;
    if( fabs(term) < fabs(sum)*EPS )
      // global coefficient e^-x * x^a / Gamma(a)
      return sum;
  }
  std::cerr << "GammaSeries::a too large, ITMAX too small" << std::endl;
  return 0.;
}

