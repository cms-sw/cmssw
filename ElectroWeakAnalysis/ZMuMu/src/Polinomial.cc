#include "ElectroWeakAnalysis/ZMuMu/interface/Polinomial.h"

Polinomial::Polinomial(double a_0, double a_1, double a_2) {
  setParameters(a_0, a_1, a_2);
}

void Polinomial::setParameters(double a_0, double a_1, double a_2) {
  a_0_ = a_0;
  a_1_ = a_1;
  a_2_ = a_2;
}

double Polinomial::operator()(double x) const {
  double poli = 0;
  poli = a_0_*x*x+a_1_*x+a_2_; //poli = l3_*(-l_*(l_*a_*x+a_+ l_)-b_*(l2_*x*x+2*l_*x+2));
  return poli;
}
