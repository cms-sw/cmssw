#ifndef ElectroWeakAnalysis_ZMuMu_ConvZShapeGauss_h
#define ElectroWeakAnalysis_ZMuMu_ConvZShapeGauss_h

#include "ElectroWeakAnalysis/ZMuMu/interface/ZetaShape.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/Gaussian.h"

class ConvZShapeGauss {
 public: 
  ConvZShapeGauss(ZetaShape & zs, Gaussian & gauss, double deltax, int bins);
  double operator()(double x) const;  
 private:
  ZetaShape zs_;
  Gaussian gauss_;
  double deltax_;
  int bins_;
};
#endif
