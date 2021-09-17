#ifndef PhysicsTools_Utilities_ZMuMu_BreitWigner_h
#define PhysicsTools_Utilities_ZMuMu_BreitWigner_h
#include "PhysicsTools/Utilities/interface/Parameter.h"

#include <cmath>

namespace funct {
  const double twoOverPi = 2. / M_PI;

  struct BreitWigner {
    BreitWigner(const Parameter& m, const Parameter& g) : mass(m.ptr()), width(g.ptr()) {}
    BreitWigner(std::shared_ptr<double> m, std::shared_ptr<double> g) : mass(m), width(g) {}
    BreitWigner(double m, double g) : mass(new double(m)), width(new double(g)) {}
    double operator()(double x) const {
      double m2 = *mass * (*mass);
      double g2 = *width * (*width);
      double g2OverM2 = g2 / m2;
      double s = x * x;
      double deltaS = s - m2;
      double lineShape = 0;
      if (fabs(deltaS / m2) < 16) {
        double prop = deltaS * deltaS + s * s * g2OverM2;
        lineShape = twoOverPi * (*width) * s / prop;
      }
      return lineShape;
    }
    std::shared_ptr<double> mass, width;
  };

}  // namespace funct

#endif
