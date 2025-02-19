/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuBackNorm.h"
#include <cmath> 

ZMuMuBackNorm::ZMuMuBackNorm(double l, double a, double b):
  expo_(-l), 
  pol_(-pow(l, 2)-l*a-2*b, -pow(l, 2)*a-2*l*b, -pow(l, 2)*b) {
} 

void ZMuMuBackNorm::setParameters(double l, double a, double b) {
  expo_.setParameters(-l);
  double l2 = l*l;
  pol_.setParameters(-l2-l*a-2*b, -l2*a-2*l*b, -l2*b);
}

double ZMuMuBackNorm::operator()(const int x_min, const int x_max) const {
  double l = - expo_.lambda; //the exponential is constructed as negative!!
  double l3inv = 1/(l*l*l);
  double N1 = expo_(x_max)*l3inv * pol_(x_max);
  double N2 = expo_(x_min)*l3inv * pol_(x_min);
  return 1/(N1 - N2);
}
*/
