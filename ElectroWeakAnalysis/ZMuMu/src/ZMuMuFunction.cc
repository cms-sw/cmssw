/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuFunction.h"

void ZMuMuFunction::setParameters(double m, double g, double Ng, double Ni, 
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

double ZMuMuFunction::operator()(double x) const {
  double eff_tr_2 = efficiencyTrack * efficiencyTrack;
  double eff_sa_2 = efficiencyStandalone * efficiencyStandalone;
  return cgz_(x) * numberOfEvents * eff_tr_2 * eff_sa_2;
}
*/
