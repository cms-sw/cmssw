/**
 Description: Function to propagate from a helix to a plane on the GPU
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_helixArbitraryPlaneCrossing2Order_h
#define RecoEgamma_EgammaElectronAlgos_interface_helixArbitraryPlaneCrossing2Order_h

#include "RecoEgamma/EgammaElectronAlgos/interface/Plane.h"
#include <cmath>
#include <cfloat>

#include "DataFormats/EgammaReco/interface/alpaka/Phys3DVector.h"

using Vec3d = cms::alpakatools::math::Phys3DVector<double>;

namespace propagators {

  namespace planeCrossing2Order {

    constexpr inline Vec3d positionInDouble(const double theRho,
                                            const double s,
                                            const double x0,
                                            const double y0,
                                            const double z0,
                                            const double cosPhi0,
                                            const double sinPhi0,
                                            const double cosTheta,
                                            const double sinThetaI) {
      const double st = s / sinThetaI;

      Vec3d res;

      res[0] = x0 + (cosPhi0 - (st * 0.5 * theRho) * sinPhi0) * st;
      res[1] = y0 + (sinPhi0 + (st * 0.5 * theRho) * cosPhi0) * st;
      res[2] = z0 + st * cosTheta * sinThetaI;

      return res;
    }

    constexpr inline Vec3d directionInDouble(const double theRho,
                                             const double s,
                                             const double cosPhi0,
                                             const double sinPhi0,
                                             const double cosTheta,
                                             const double sinThetaI) {
      const double dph = s * theRho / sinThetaI;

      Vec3d res;

      res[0] = cosPhi0 - (sinPhi0 + 0.5 * dph * cosPhi0) * dph;
      res[1] = sinPhi0 + (cosPhi0 - 0.5 * dph * sinPhi0) * dph;
      res[2] = cosTheta * sinThetaI;

      return res;
    }

    template <typename TAcc, PropagationDirection propDir>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool solutionByDirection(TAcc const& acc,
                                                                 double& path,  //0
                                                                 const double dS1,
                                                                 const double dS2) {
      bool valid = false;

      if constexpr (propDir == PropagationDirection::anyDirection) {
        valid = true;
        path = (alpaka::math::abs(acc, dS1) < alpaka::math::abs(acc, dS2)) ? dS1 : dS2;
      } else {
        constexpr double propSign = (propDir == PropagationDirection::alongMomentum) ? 1. : -1.;

        double s1(propSign * dS1);
        double s2(propSign * dS2);

        path = 0.;

        if (s1 > s2) {
          //std::swap(s1, s2);
          const double tmp = s1;
          s1 = s2;
          s2 = tmp;
        }
        if ((s1 < 0) && (s2 >= 0)) {
          valid = true;
          path = propSign * s2;
        } else if (s1 >= 0) {
          valid = true;
          path = propSign * s1;
        }
      }

      if (!(alpaka::math::isfinite(acc, path)))
        valid = false;

      return valid;
    }
  };  // namespace planeCrossing2Order

  template <typename TAcc, PropagationDirection propDir>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixArbitraryPlaneCrossing2Order(
      TAcc const& acc,
      const Vec3d& point,
      const Vec3d& direction,
      const float curvature,
      const egamma::Plane<typename Vec3d::value_type> plane,
      double& pathLength,
      bool& validPath,
      Vec3d& position,
      Vec3d& directionOut) {
    const double theX0 = point[0];
    const double theY0 = point[1];
    const double theZ0 = point[2];
    const double px = direction[0];
    const double py = direction[1];
    const double pz = direction[2];
    const double pt2 = px * px + py * py;
    const double p2 = pt2 + pz * pz;
    const double pI = 1.0 / alpaka::math::sqrt(acc, p2);
    const double ptI = 1.0 / alpaka::math::sqrt(acc, pt2);
    const double theCosPhi0 = px * ptI;
    const double theSinPhi0 = py * ptI;
    const double theCosTheta = pz * pI;
    const double theSinThetaI = pt2 * ptI * pI;

    // Get normal vector of the plane
    const Vec3d normalToPlane = plane.normalVector();

    const double nPx = normalToPlane[0];
    const double nPy = normalToPlane[1];
    const double nPz = normalToPlane[2];
    const double cP = plane.localZ(point);

    // Coefficients of the 2nd order equation
    const double ceq1 = curvature * (nPx * theSinPhi0 - nPy * theCosPhi0);
    const double ceq2 = nPx * theCosPhi0 + nPy * theSinPhi0 + nPz * theCosTheta * theSinThetaI;
    const double ceq3 = cP;

    //
    // Check for degeneration to linear equation (zero
    // curvature, forward plane or direction perp. to plane)
    //

    double dS1, dS2;
    if (alpaka::math::abs(acc, ceq1) > FLT_MIN) {
      const double deq1 = ceq2 * ceq2;
      const double deq2 = ceq1 * ceq3;
      if (alpaka::math::abs(acc, deq1) < FLT_MIN || alpaka::math::abs(acc, deq2 / deq1) > 1.e-6) {
        //
        // Standard solution for quadratic equations
        //
        const double deq = deq1 + 2 * deq2;
        if (deq < 0.) {
          validPath = false;
          return;
        }
        const double ceq = ceq2 + alpaka::math::copysign(acc, alpaka::math::sqrt(acc, deq), ceq2);
        dS1 = (ceq / ceq1) * theSinThetaI;
        dS2 = -2. * (ceq3 / ceq) * theSinThetaI;
      } else {
        const double ceq = (ceq2 / ceq1) * theSinThetaI;
        double deq = deq2 / deq1;
        deq *= (1 - 0.5 * deq);
        dS1 = -ceq * deq;
        dS2 = ceq * (2 + deq);
      }
    } else {
      //
      // Special case: linear equation
      //
      dS1 = dS2 = -(ceq3 / ceq2) * theSinThetaI;
    }

    // Choose solution based on direction
    validPath = planeCrossing2Order::solutionByDirection<TAcc, propDir>(acc, pathLength, dS1, dS2);

    if (validPath) {
      // Calculate position and direction
      position = planeCrossing2Order::positionInDouble(
          curvature, pathLength, theX0, theY0, theZ0, theCosPhi0, theSinPhi0, theCosTheta, theSinThetaI);
      directionOut = planeCrossing2Order::directionInDouble(
          curvature, pathLength, theCosPhi0, theSinPhi0, theCosTheta, theSinThetaI);
    }
  }

}  // namespace propagators

#endif  // RecoEgamma_EgammaElectronAlgos_interface_helixArbitraryPlaneCrossing2Order_h
