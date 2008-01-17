#include "PhysicsTools/Utilities/interface/Exponential.h"
#include <cmath>

Exponential::Exponential(double lambda) {
  setParameters(lambda);
}

void Exponential::setParameters(double lambda) {
  lambda_ = lambda;
}

double Exponential::operator()(double x) const {
  return exp(lambda_*x);
}
