#include "ElectroWeakAnalysis/ZMuMu/interface/Exponential.h"
#include <cmath>

Exponential::Exponential(double parameter) {
  setParameters(parameter);
}

void Exponential::setParameters(double parameter) {
  parameter_ = parameter;
}

double Exponential::operator()(double x) const {
  double expo = 0;
  expo = exp(parameter_*x);
  return expo;
}
