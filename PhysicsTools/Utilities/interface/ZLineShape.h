#ifndef PhysicsTools_Utilities_ZLineShape_h
#define PhysicsTools_Utilities_ZLineShape_h
#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/GammaPropagator.h"
#include "PhysicsTools/Utilities/interface/GammaZInterference.h"
#include <boost/shared_ptr.hpp>

namespace function {
  class ZLineShape {
  public:
    enum { arguments = 1 };
    enum { parameters = 4 };
    ZLineShape(boost::shared_ptr<double> m, 
	       boost::shared_ptr<double> g, 
	       boost::shared_ptr<double> Ng, 
	       boost::shared_ptr<double> Ni)
      : mass(m), width(g), Ngamma(Ng), Nint(Ni), bw_(m, g), gp_(), gzi_(m, g) {}
    ZLineShape(double m, double g, double Ng, double Ni)
      : mass(new double(m)), width(new double(g)), Ngamma(new double(Ng)), Nint(new double(Ni)), bw_(m, g), gp_(), gzi_(m, g) {}
    void setParameters(double m, double g, double Ng, double Ni) {
      *mass = m;
      *width = g;
      *Ngamma = Ng;
      *Nint = Ni;
      bw_.setParameters(m, g);
      gp_.setParameters();
      gzi_.setParameters(m, g);
    }
    double operator()(double x) const {
      return (1.0 - *Nint - *Ngamma) * bw_(x) + *Ngamma * gp_(x) + *Nint * gzi_(x);
  }
    boost::shared_ptr<double> mass, width, Ngamma, Nint; 
  private:
    BreitWigner bw_;
    GammaPropagator gp_;
    GammaZInterference gzi_;
  };

}

#endif

