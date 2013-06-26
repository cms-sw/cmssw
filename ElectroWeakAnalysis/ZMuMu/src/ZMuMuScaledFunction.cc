/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuScaledFunction.h"

ZMuMuScaledFunction::ZMuMuScaledFunction(const ZMuMuFunction & ZMM, int bin):
  ZMM_(ZMM), bin_(bin) {
}

ZMuMuScaledFunction::ZMuMuScaledFunction(double m, double g, double Ng, double Ni, 
					 double me, double s, 
					 double N, double eff_tr, double eff_sa, int bin):
  ZMM_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa), bin_(bin) {
}

void ZMuMuScaledFunction::setParameters(double m, double g, double Ng, double Ni, 
					double me, double s, 
					double N, double eff_tr, double eff_sa) { 
  ZMM_.setParameters(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa);
}


double ZMuMuScaledFunction::operator()(double x) const {
  return bin_ * ZMM_(x);
}
*/
