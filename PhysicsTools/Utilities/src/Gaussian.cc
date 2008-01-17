#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include <cmath>

const double oneOverSqrtTwoPi = 1/sqrt(2*M_PI);

Gaussian::Gaussian(double mean, double sigma) :
  mean_(mean), sigma_(sigma) {
}

void Gaussian::setParameters(double mean, double sigma) {
  mean_ = mean;
  sigma_ = sigma;
}
  
double Gaussian::operator()(double x) const {
  double z = (x-mean_)/sigma_;
  if(fabs(z)>8) return 0;
  return oneOverSqrtTwoPi/sigma_*exp(-z*z/2);
}
