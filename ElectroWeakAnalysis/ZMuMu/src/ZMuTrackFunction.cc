/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuTrackFunction.h"

void ZMuTrackFunction::setParameters(double m, double g, double Ng, double Ni, 
				     double me, double s, 
				     double N, double eff_tr, double eff_sa) {
  mass = m; 
  width = g; 
  Ngamma = Ng; 
  Nint = Ni; 
  mean = me; 
  sigma = s; 
  numberOfEvents = N; 
  efficiencyTrack = eff_tr; 
  efficiencyStandalone = eff_sa; 
  cgz_.setParameters(m, g, Ng, Ni, me, s);
}

double ZMuTrackFunction::operator()(double x) const {
  double eff_tr_2 = efficiencyTrack * efficiencyTrack;
  double eff_sa_minus = efficiencyStandalone * (1. - efficiencyStandalone);
  return cgz_(x) * 2. * numberOfEvents * eff_tr_2 * eff_sa_minus;
}
*/
