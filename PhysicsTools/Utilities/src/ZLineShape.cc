#include "PhysicsTools/Utilities/interface/ZLineShape.h"

ZLineShape::ZLineShape(double m, double g, double Nf, double Ni) :
  bw_(m, g), gp_(), gzi_(m, g) {
  Nf_ = Nf;
  Ni_ = Ni;
}

void ZLineShape::setParameters(double m, double g, double Nf, double Ni) {
  Nf_ = Nf;
  Ni_ = Ni;
  bw_.setParameters(m, g);
  gp_.setParameters();
  gzi_.setParameters(m, g);
}

double ZLineShape::operator()(double x) const {
  double zetashape = (1.0 - Ni_ - Nf_)*bw_(x) + Nf_*gp_(x) + Ni_*gzi_(x);
  return zetashape;
}
