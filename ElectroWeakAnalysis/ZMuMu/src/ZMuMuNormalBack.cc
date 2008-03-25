/*
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuNormalBack.h"

ZMuMuNormalBack::ZMuMuNormalBack(double Nb, double l, double a, double b, 
				 int x_min, int x_max):
  zmb_(Nb, l, a, b), 
  zmbn_(l, a, b), 
  x_min_(x_min), x_max_(x_max) {} 

void ZMuMuNormalBack::setParameters(double Nb, double l, double a, double b) { 
  zmb_.setParameters(Nb, l, a, b);
  zmbn_.setParameters(l, a, b);
} 

double ZMuMuNormalBack::operator()(double x) const { 
  return zmbn_(x_min_, x_max_) * zmb_(x);
}
*/
