#ifndef PhysicsTools_Utilities_GammaZInterference_h
#define PhysicsTools_Utilities_GammaZInterference_h
#include <boost/shared_ptr.hpp>

namespace function {

  struct GammaZInterference {
    enum { arguments = 1 };
    enum { parameters = 2 };
    GammaZInterference(boost::shared_ptr<double> m, boost::shared_ptr<double> g): mass(m), width(g) {}
    GammaZInterference(double m, double g): mass(new double(m)), width(new double(g)) {}
    void setParameters(double m, double g) { 
      *mass = m;
      *width = g;
    }
    double operator()(double x) const { 
      double m2 = *mass * (*mass); 
      double g2 = *width * (*width);
      double g2OverM2 = g2/m2; 
      double s = x*x;
      double deltaS = s - m2;
      double interference = 0;
      if (fabs(deltaS/m2)<16) {
	double prop = deltaS*deltaS + s*s*g2OverM2;
	interference =  5*(*mass)*deltaS/prop;
      }
      return interference;
    }
    boost::shared_ptr<double> mass, width;
  };

}

#endif
