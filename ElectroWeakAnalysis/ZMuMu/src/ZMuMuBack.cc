/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuBack.h"

void ZMuMuBack::setParameters(double Nb, double l, double a, double b) {
  *Nbkg = Nb; 
  *lambda = l;
  *a1 = a;
  *a2 = b;
  expo_.setParameters(-l);
  poly_.setParameters(1., a, b);
}

double ZMuMuBack::operator()(double x) const {
  return *Nbkg * expo_(x) * poly_(x);
}
*/
