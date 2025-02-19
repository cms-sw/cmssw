#include "DataFormats/Math/interface/normalizedPhi.h"
#include <cmath>

double normalizedPhi(double phi) {
  static double const TWO_PI = M_PI * 2;
  while ( phi < -M_PI ) phi += TWO_PI;
  while ( phi >  M_PI ) phi -= TWO_PI;
  return phi;
}
