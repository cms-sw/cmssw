#ifndef PhysicsTools_Utilities_Exponential_h
#define PhysicsTools_Utilities_Exponential_h
#include "PhysicsTools/Utilities/interface/Parameter.h"

#include <cmath>

namespace funct {

  struct Exponential {
    Exponential(const Parameter& l) : lambda(l.ptr()) {}
    Exponential(std::shared_ptr<double> l) : lambda(l) {}
    Exponential(double l) : lambda(new double(l)) {}
    double operator()(double x) const { return std::exp((*lambda) * x); }
    std::shared_ptr<double> lambda;
  };

}  // namespace funct

#endif
