#ifndef ElectroWeakAnalysis_ZMuMu_ZMuMuBackNorm_h
#define ElectroWeakAnalysis_ZMuMu_ZMuMuBackNorm_h

#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Polynomial.h"
#include <boost/shared_ptr.hpp>
#include <cmath> 

namespace function {
  class ZMuMuBackNorm {
   public:
    enum{ arguments = 2 };
    enum{ parameters = 3 };
    ZMuMuBackNorm(boost::shared_ptr<double> l, boost::shared_ptr<double> a, boost::shared_ptr<double> b):
      lambda(l), a1(a), a2(b), 
      expo_(-(*l)), 
      pol_(-pow(*l, 2)- (*l) * (*a) - 2 * (*b), -pow(*l, 2) * (*a) - 2 * (*l) * (*b), -pow(*l, 2) * (*b)) {}
    ZMuMuBackNorm(double l, double a, double b): 
      lambda(new double(l)), a1(new double(a)), a2(new double(b)), 
      expo_(-l), pol_(-pow(l, 2)-l*a-2*b, -pow(l, 2)*a-2*l*b, -pow(l, 2)*b) {}
    ZMuMuBackNorm(const Exponential & expo, const Polynomial<2> & pol) : expo_(expo), pol_(pol) { }
    double operator()(const int x_min, const int x_max) const {
      double l = - (*(expo_.lambda)); //the exponential is constructed as negative!!
      double l3inv = 1/(l*l*l);
      double N1 = expo_(x_max)*l3inv * pol_(x_max);
      double N2 = expo_(x_min)*l3inv * pol_(x_min);
      return 1/(N1 - N2);
    }
    void setParameters(double l, double a, double b) { 
      expo_.setParameters(-l);
      double l2 = l*l;
      pol_.setParameters(-l2-l*a-2*b, -l2*a-2*l*b, -l2*b);
    }
    boost::shared_ptr<double> lambda, a1, a2;
  private:
    Exponential expo_;
    Polynomial<2> pol_; 
  };
}

#endif
