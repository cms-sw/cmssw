#ifndef PhysicsTools_Utilities_ZLineShape_h
#define PhysicsTools_Utilities_ZLineShape_h
#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/GammaPropagator.h"
#include "PhysicsTools/Utilities/interface/GammaZInterference.h"

class ZLineShape {
 public:
  ZLineShape(double m, double g, double Nf, double Ni);
  void setParameters(double m, double g, double Nf, double Ni);
  double operator()(double x) const;

 private:
  BreitWigner bw_;
  GammaPropagator gp_;
  GammaZInterference gzi_;
  double Ni_, Nf_;
};

#endif

