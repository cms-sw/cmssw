#include "ElectroWeakAnalysis/ZMuMu/interface/Gaussian.h"
#include <cmath>

const double oneOverSqrtTwoPi = 1/sqrt(2*M_PI);

Gaussian::Gaussian(double mean, double sigma) {
  setParameters(mean, sigma);
}

void Gaussian::setParameters(double mean, double sigma) {
  mean_ = mean;
  sigma_ = sigma;
  mean2_ = mean*mean; 
  sigma2_ = sigma*sigma;
}
  
double Gaussian::operator()(double x) const {
  double gauss = 0;
  gauss = oneOverSqrtTwoPi/sigma_*exp(-pow(x-mean_, 2.)/(2*sigma2_));
  return gauss;
}

double Gaussian::confidenceLevel99_7Interval() const {
  return 3*sigma_;
}
