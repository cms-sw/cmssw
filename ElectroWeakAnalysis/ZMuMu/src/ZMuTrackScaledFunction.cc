/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuTrackScaledFunction.h"

ZMuTrackScaledFunction::ZMuTrackScaledFunction(const ZMuTrackFunction & ZMM, int bin):
  ZMM_(ZMM), bin_(bin) {
}

ZMuTrackScaledFunction::ZMuTrackScaledFunction(double m, double g, double Ng, double Ni, 
					       double me, double s, 
					       double N, double eff_tr, double eff_sa, int bin):
  ZMM_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa), bin_(bin) {
}

void ZMuTrackScaledFunction::setParameters(double m, double g, double Ng, double Ni, 
					   double me, double s, 
					   double N, double eff_tr, double eff_sa) { 
  ZMM_.setParameters(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa);
}

void ZMuTrackScaledFunction::setConstants(int bin) { 
  bin_ = bin;
}

double ZMuTrackScaledFunction::operator()(double x) const {
  return bin_ * ZMM_(x);
}
*/

