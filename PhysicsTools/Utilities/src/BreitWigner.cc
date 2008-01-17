#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include <cmath>

const double twoOverPi = 2./M_PI;

BreitWigner::BreitWigner(double m, double g ){
  setParameters(m, g);
}

void BreitWigner::setParameters(double m, double g){
  m_ = m;
  g_ = g;
  m2_ = m*m; 
  g2_ = g*g;
  g2OverM2_ = g2_/m2_;
}

double BreitWigner::operator()(double x) const{
  double s = x*x;
  double deltaS = s - m2_;
  double lineShape = 0;
  if (fabs(deltaS/m2_)<16) {
    double prop = deltaS*deltaS + s*s*g2OverM2_;
    lineShape =  twoOverPi*g_*s/prop;
  }
  return lineShape;
}
