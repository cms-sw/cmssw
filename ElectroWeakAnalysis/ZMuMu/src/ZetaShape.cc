#include "ElectroWeakAnalysis/ZMuMu/interface/ZetaShape.h"

ZetaShape::ZetaShape(double m, double g, double Nf, double Ni) :
  bw_(m, g), gp_(), gzi_(m, g) {
  Nf_ = Nf;
  Ni_ = Ni;
}

void ZetaShape::setParameters(double m, double g, double Nf, double Ni) {
  Nf_ = Nf;
  Ni_ = Ni;
  bw_.setParameters(m, g);
  gp_.setParameters();
  gzi_.setParameters(m, g);
}

double ZetaShape::operator()(double x) const {
  double zetashape = (1.0 - Ni_ - Nf_)*bw_(x) + Nf_*gp_(x) + Ni_*gzi_(x);
  return zetashape;
}
