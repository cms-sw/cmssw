#ifndef ElectroWeakAnalysis_ZMuMu_ZetaShape_h
#define ElectroWeakAnalysis_ZMuMu_ZetaShape_h
#include "ElectroWeakAnalysis/ZMuMu/interface/BreitWigner.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/GammaPropagator.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/GammaZetaInterference.h"

class ZetaShape {
 public:
  ZetaShape(double m, double g, double Nf, double Ni);
  void setParameters(double m, double g, double Nf, double Ni);
  double operator()(double x) const;

 private:
  BreitWigner bw_;
  GammaPropagator gp_;
  GammaZetaInterference gzi_;
  double Ni_, Nf_;
};

#endif

