/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuStandaloneScaledFunction.h"

ZMuStandaloneScaledFunction::ZMuStandaloneScaledFunction(const ZMuStandaloneFunction & ZMM, int bin):
  ZMM_(ZMM), bin_(bin) {
}

ZMuStandaloneScaledFunction::ZMuStandaloneScaledFunction(double m, double g, double Ng, double Ni, 
							 double me, double s, 
							 double N, double eff_tr, double eff_sa, int bin):
  ZMM_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa), bin_(bin) {
}

void ZMuStandaloneScaledFunction::setParameters(double m, double g, double Ng, double Ni, 
						double me, double s, 
						double N, double eff_tr, double eff_sa) { 
  ZMM_.setParameters(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa);
}

void ZMuStandaloneScaledFunction::setConstants(int bin) { 
  bin_ = bin;
}

double ZMuStandaloneScaledFunction::operator()(double x) const {
  return bin_ * ZMM_(x);
}
*/
