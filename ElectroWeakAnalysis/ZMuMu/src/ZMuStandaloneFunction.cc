/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuStandaloneFunction.h"

void ZMuStandaloneFunction::setParameters(double m, double g, double Ng, double Ni, 
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


double ZMuStandaloneFunction::operator()(double x) const {
  double eff_sa_2 = efficiencyStandalone * efficiencyStandalone;
  double eff_tr_minus = efficiencyTrack * ( 1. - efficiencyTrack );
  return cgz_(x) * 2. * numberOfEvents * eff_sa_2 * eff_tr_minus;
}
*/
