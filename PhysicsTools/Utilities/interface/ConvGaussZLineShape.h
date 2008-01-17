#ifndef PhysicsTools_Utilities_ConvGaussZLineShape_h
#define PhysicsTools_Utilities_ConvGaussZLineShape_h

#include "PhysicsTools/Utilities/interface/ZLineShape.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"

class ConvGaussZLineShape {
 public: 
  ConvGaussZLineShape(double m, double g, double Nf, double Ni, 
		      double mean, double sigma, double deltax, int bins);
  double operator()(double x) const;  
  void setParameters(double m, double g, double Nf, double Ni, 
		     double mean, double sigma, double deltax, int bins);
 private:
  ZLineShape zs_;
  Gaussian gauss_;
  double deltax_;
  int bins_;
  double dx_;
};
#endif
