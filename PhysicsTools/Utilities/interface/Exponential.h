#ifndef PhysicsTools_Utilities_Exponential_h
#define PhysicsTools_Utilities_Exponential_h
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include <boost/shared_ptr.hpp>
#include <cmath>

namespace funct {

  struct Exponential {
    Exponential(const Parameter & l) : lambda(l.ptr()) { }
    Exponential(boost::shared_ptr<double> l) : lambda(l) { }
    Exponential(double l) : lambda(new double(l)) { }
    double operator()(double x) const { return std::exp((*lambda)*x); }
    boost::shared_ptr<double> lambda;
  };

}

#endif
