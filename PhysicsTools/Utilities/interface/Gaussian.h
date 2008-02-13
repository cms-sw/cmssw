#ifndef PhysicsTools_Utilities_Gaussian_h
#define PhysicsTools_Utilities_Gaussian_h
#include <boost/shared_ptr.hpp>
#include <cmath>

namespace function {
  const double oneOverSqrtTwoPi = 1/sqrt(2*M_PI);
  
  struct Gaussian {
    enum { arguments = 1 };
    enum { parameters = 2 };
    Gaussian(boost::shared_ptr<double> m, boost::shared_ptr<double> s): mean(m), sigma(s) { }
    Gaussian(double m, double s): mean(new double(m)), sigma(new double(s)){}
    Gaussian(const Gaussian& g) : mean(g.mean), sigma(g.sigma) { }
    void setParameters(double m, double s){
      *mean = m;
      *sigma = s;
    }
    double operator()(double x) const {
      double z = (x - *mean)/ *sigma;
      if(fabs(z)>8) return 0;
      return oneOverSqrtTwoPi/ *sigma * exp(-z*z/2);
    }
    boost::shared_ptr<double> mean, sigma;
  };

}

#endif
