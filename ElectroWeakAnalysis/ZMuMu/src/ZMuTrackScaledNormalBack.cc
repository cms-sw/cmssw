/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuTrackScaledNormalBack.h"

ZMuTrackScaledNormalBack::ZMuTrackScaledNormalBack(double m, double g, double Ng, double Ni, 
						   double me, double s, 
						   double N, double eff_tr, double eff_sa, 
						   double Nb, double l, double a, double b, 
						   int bin, int x_min, int x_max):
  zmt_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa), 
  zmb_(Nb, l, a, b), 
  zmbn_(l, a, b), 
  bin_(bin), x_min_(x_min), x_max_(x_max) {} 

void ZMuTrackScaledNormalBack::setParameters(double m, double g, double Ng, double Ni, 
					     double me, double s, 
					     double N, double eff_tr, double eff_sa, 
					     double Nb, double l, double a, double b) { 
  zmt_.setParameters(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa);
  zmb_.setParameters(Nb, l, a, b);
  zmbn_.setParameters(l, a, b);
} 

void ZMuTrackScaledNormalBack::setConstants(int bin, int x_min, int x_max) {
  bin_ = bin; 
  x_min_ = x_min; 
  x_max_ = x_max;
}

double ZMuTrackScaledNormalBack::operator()(double x) const { 
  return bin_ * zmt_(x) + zmbn_(x_min_, x_max_) * zmb_(x);
}
*/
