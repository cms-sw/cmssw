
#ifndef PhysicsTools_Utilities_ConvGaussZLineShape_h
#define PhysicsTools_Utilities_ConvGaussZLineShape_h
#include "PhysicsTools/Utilities/interface/ZLineShape.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include <boost/shared_ptr.hpp>

namespace function {
  class ConvGaussZLineShape {
  public:
    enum { arguments = 1 };
    enum { parameters = 6 };
    ConvGaussZLineShape(const Parameter& m, const Parameter& g, 
			const Parameter& Ng, const Parameter& Ni, 
			const Parameter& me, const Parameter& s):
      mass(m.ptr()), width(g.ptr()), 
      Ngamma(Ng.ptr()), Nint(Ni.ptr()), 
      mean(me.ptr()), sigma(s.ptr()), 
      zs_(m, g, Ng, Ni), gauss_(me, s) {}
    ConvGaussZLineShape(boost::shared_ptr<double> m, boost::shared_ptr<double> g, 
			boost::shared_ptr<double> Ng, boost::shared_ptr<double> Ni, 
			boost::shared_ptr<double> me, boost::shared_ptr<double> s) : 
      mass(m), width(g), Ngamma(Ng), Nint(Ni), mean(me), sigma(s), 
      zs_(m, g, Ng, Ni), gauss_(me, s) {}
    double operator()(double x) const;  
    boost::shared_ptr<double> mass, width, Ngamma, Nint, mean, sigma;
  private:
    ZLineShape zs_;
    Gaussian gauss_;
  };
}

#endif

