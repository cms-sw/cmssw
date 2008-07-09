#ifndef PhysicsTools_Utilities_ZLineShape_h
#define PhysicsTools_Utilities_ZLineShape_h
#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/GammaPropagator.h"
#include "PhysicsTools/Utilities/interface/GammaZInterference.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include <boost/shared_ptr.hpp>

namespace funct {
  class ZLineShape {
  public:
    ZLineShape(const Parameter& m, const Parameter& g, 
	       const Parameter& Ng, const Parameter& Ni): 
      Ngamma(Ng.ptr()), Nint(Ni.ptr()), 
      bw_(m, g), gp_(), gzi_(m, g) {}
    ZLineShape(boost::shared_ptr<double> m, boost::shared_ptr<double> g, 
	       boost::shared_ptr<double> Ng, boost::shared_ptr<double> Ni): 
      Ngamma(Ng), Nint(Ni), 
      bw_(m, g), gp_(), gzi_(m, g) {}
    double operator()(double x) const {
      return (1.0 - *Nint - *Ngamma) * bw_(x) + *Ngamma * gp_(x) + *Nint * gzi_(x);
    }
    boost::shared_ptr<double> Ngamma, Nint; 
  private:
    BreitWigner bw_;
    GammaPropagator gp_;
    GammaZInterference gzi_;
  };

}

#endif

