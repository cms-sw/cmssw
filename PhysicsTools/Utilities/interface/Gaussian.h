#ifndef PhysicsTools_Utilities_Gaussian_h
#define PhysicsTools_Utilities_Gaussian_h
#include "PhysicsTools/Utilities/interface/Parameter.h"

#include <cmath>
#include <utility>

namespace funct {

  const double oneOverSqrtTwoPi = 1 / sqrt(2 * M_PI);

  struct Gaussian {
    Gaussian(const Parameter& m, const Parameter& s) : mean(m.ptr()), sigma(s.ptr()) {}
    Gaussian(std::shared_ptr<double> m, std::shared_ptr<double> s) : mean(std::move(m)), sigma(std::move(s)) {}
    Gaussian(double m, double s) : mean(new double(m)), sigma(new double(s)) {}
    double operator()(double x) const {
      double z = (x - *mean) / *sigma;
      if (fabs(z) > 8)
        return 0;
      return oneOverSqrtTwoPi / *sigma * exp(-z * z / 2);
    }
    std::shared_ptr<double> mean, sigma;
  };

}  // namespace funct

#endif
