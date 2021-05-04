#ifndef PhysicsTools_Utilities_GammaZInterference_h
#define PhysicsTools_Utilities_GammaZInterference_h
#include "PhysicsTools/Utilities/interface/Parameter.h"

#include <cmath>

namespace funct {

  struct GammaZInterference {
    GammaZInterference(const Parameter& m, const Parameter& g) : mass(m.ptr()), width(g.ptr()) {}
    GammaZInterference(std::shared_ptr<double> m, std::shared_ptr<double> g) : mass(m), width(g) {}
    double operator()(double x) const {
      double m2 = *mass * (*mass);
      double g2 = *width * (*width);
      double g2OverM2 = g2 / m2;
      double s = x * x;
      double deltaS = s - m2;
      double interference = 0;
      if (fabs(deltaS / m2) < 16) {
        double prop = deltaS * deltaS + s * s * g2OverM2;
        interference = 5 * (*mass) * deltaS / prop;
      }
      return interference;
    }
    std::shared_ptr<double> mass, width;
  };

}  // namespace funct

#endif
