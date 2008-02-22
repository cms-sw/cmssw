#include "ElectroWeakAnalysis/ZMuMu/interface/GammaZetaInterference.h"
#include <cmath>

GammaZetaInterference::GammaZetaInterference(double m, double g){
  setParameters(m, g);
}

void GammaZetaInterference::setParameters(double m, double g){
  m_ = m;
  g_ = g;
  m2_ = m*m; 
  g2_ = g*g;
  g2OverM2_ = g2_/m2_;
}
  
double GammaZetaInterference::operator()(double x) const{
  double s = x*x;
  double deltaS = s - m2_;
  double interference = 0;
  if (fabs(deltaS/m2_)<16) {
    double prop = deltaS*deltaS + s*s*g2OverM2_;
    interference =  5*m_*deltaS/prop;
  }
  return interference;
}
