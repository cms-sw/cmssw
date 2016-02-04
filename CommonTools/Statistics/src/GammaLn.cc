#include "CommonTools/Statistics/src/GammaLn.h"
#include <cmath>

float
GammaLn( float z )
{
  const static double coefficients[6] = 
  { 76.18009172947146, -86.50532032941677,      24.01409824083091,
    -1.231739572450155,  0.1208650973866179e-2, -0.5395239384953e-5 };

  double temp = z+5.5;
  temp -= (z+0.5)*log(temp);
  double y = z;
  double series = 1.000000000190015;
  for( int term = 0; term < 6; term++ )
    series += coefficients[term]/++y;
  return -temp + log(2.5066282746310005*series/z);
}
