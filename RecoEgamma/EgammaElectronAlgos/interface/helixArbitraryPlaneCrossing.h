/**
 Description: Function to propagate from a point to a plane on the GPU
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_helixArbitraryPlaneCrossing_h
#define RecoEgamma_EgammaElectronAlgos_interface_helixArbitraryPlaneCrossing_h

#include <alpaka/alpaka.hpp>
#include <cmath>
#include <utility>
#include <iostream>
#include <atomic>

#include "RecoEgamma/EgammaElectronAlgos/interface/helixArbitraryPlaneCrossing2Order.h"

namespace propagators {

  constexpr float theNumericalPrecision = 5.e-7f;
  constexpr float theMaxDistToPlane = 1.e-4f;

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vec3d positionInDouble(TAcc const& acc,
                                                             const double s,
                                                             const Vec3d& point,
                                                             const double rho,
                                                             const double cosPhi0,
                                                             const double sinPhi0,
                                                             const double cosTheta,
                                                             const double sinTheta,
                                                             const double sinThetaI,
                                                             double& theCachedS,
                                                             double& theCachedDPhi,
                                                             double& theCachedSDPhi,
                                                             double& theCachedCDPhi) {
    Vec3d res;

    if (s != theCachedS) {
      theCachedS = s;
      theCachedDPhi = theCachedS * rho * sinTheta;
      theCachedSDPhi = alpaka::math::sin(acc, theCachedDPhi);
      theCachedCDPhi = alpaka::math::cos(acc, theCachedDPhi);
    }

    if (alpaka::math::abs(acc, theCachedDPhi) > 1.e-4) {
      // "standard" helix formula
      const double o = 1. / rho;
      res[0] = point[0] + (-sinPhi0 * (1.0 - theCachedCDPhi) + cosPhi0 * theCachedSDPhi) * o;
      res[1] = point[1] + (cosPhi0 * (1.0 - theCachedCDPhi) + sinPhi0 * theCachedSDPhi) * o;
      res[2] = point[2] + s * cosTheta;
    } else {
      const double st = theCachedS / sinThetaI;
      res[0] = point[0] + (cosPhi0 - (st * 0.5 * rho) * sinPhi0) * st;
      res[1] = point[1] + (sinPhi0 + (st * 0.5 * rho) * cosPhi0) * st;
      res[2] = point[2] + st * cosTheta * sinThetaI;
    }
    return res;
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vec3d directionInDouble(TAcc const& acc,
                                                              const double s,
                                                              const Vec3d& point,
                                                              const double rho,
                                                              const double cosPhi0,
                                                              const double sinPhi0,
                                                              const double cosTheta,
                                                              const double sinTheta,
                                                              const double sinThetaI,
                                                              double& theCachedS,
                                                              double& theCachedDPhi,
                                                              double& theCachedSDPhi,
                                                              double& theCachedCDPhi) {
    Vec3d res;

    //
    // Calculate delta phi (if not already available)
    //

    if (s != theCachedS) {  // very very unlikely!
      theCachedS = s;
      theCachedDPhi = theCachedS * rho * sinTheta;
      theCachedSDPhi = alpaka::math::sin(acc, theCachedDPhi);
      theCachedCDPhi = alpaka::math::cos(acc, theCachedDPhi);
    }

    if (alpaka::math::abs(acc, theCachedDPhi) > 1.e-4) {
      // full helix formula
      res[0] = cosPhi0 * theCachedCDPhi - sinPhi0 * theCachedSDPhi;
      res[1] = sinPhi0 * theCachedCDPhi + cosPhi0 * theCachedSDPhi;
      res[2] = cosTheta / sinTheta;
    } else {
      // 2nd order
      const double dph = s * rho / sinThetaI;
      res[0] = cosPhi0 - (sinPhi0 + 0.5 * dph * cosPhi0) * dph;
      res[1] = sinPhi0 + (cosPhi0 - 0.5 * dph * sinPhi0) * dph;
      res[2] = cosTheta * sinThetaI;
    }
    return res;
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool notAtSurface(TAcc const& acc,
                                                        const egamma::Plane<typename Vec3d::value_type>& plane,
                                                        const Vec3d& point,
                                                        const float maxDist) {
    const float dz = static_cast<float>(plane.localZ(point));
    return alpaka::math::abs(acc, dz) > maxDist;
  }

  template <typename TAcc, PropagationDirection propDir>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixArbitraryPlaneCrossing(
      TAcc const& acc,
      const Vec3d& point,
      const Vec3d& direction,
      const float curvature,
      const egamma::Plane<typename Vec3d::value_type> plane,
      double& pathLength,
      Vec3d& position,
      Vec3d& dir,
      bool& solExists) {
    double theCachedS = 0.;
    double theCachedDPhi = 0.;
    double theCachedSDPhi = 0.;
    double theCachedCDPhi = 1.;

    constexpr int maxIterations = 20;

    const double maxNumDz = theNumericalPrecision * plane.pos_norm(acc);
    const double safeMaxDist = (theMaxDistToPlane > maxNumDz) ? theMaxDistToPlane : maxNumDz;

    const double px = direction[0];
    const double py = direction[1];
    const double pz = direction[2];
    const double pt2 = px * px + py * py;
    const double p2 = pt2 + pz * pz;
    const double pI = 1.0 / alpaka::math::sqrt(acc, p2);
    const double ptI = 1.0 / alpaka::math::sqrt(acc, pt2);
    const double cosPhi0 = px * ptI;
    const double sinPhi0 = py * ptI;
    const double cosTheta = pz * pI;
    const double sinTheta = pt2 * ptI * pI;
    const double sinThetaI = p2 * pI * ptI;  //  (1/(pt/p)) = p/pt = p*ptI and p = p2/p = p2*pI

    //
    // Prepare internal value of the propagation direction and position / direction vectors for iteration
    //

    const double dz = plane.localZ(point);
    if (alpaka::math::abs(acc, dz) < safeMaxDist) {
      pathLength = 0.0;
      solExists = true;
      position = point;
      dir = direction;

      return;
    }

    // Use existing 2nd order object at first pass
    double pathLength2O = 0;
    bool validPath2O = false;

    Vec3d position2O(0.);
    Vec3d directionOut2O(0.);

    helixArbitraryPlaneCrossing2Order<TAcc, propDir>(
        acc, point, direction, curvature, plane, pathLength2O, validPath2O, position2O, directionOut2O);

    if (!validPath2O) {
      solExists = false;
      pathLength = pathLength2O;
      return;
    }

    Vec3d xnew = positionInDouble(acc,
                                  pathLength2O,
                                  point,
                                  curvature,
                                  cosPhi0,
                                  sinPhi0,
                                  cosTheta,
                                  sinTheta,
                                  sinThetaI,
                                  theCachedS,
                                  theCachedDPhi,
                                  theCachedSDPhi,
                                  theCachedCDPhi);

    auto currentPropDir = propDir;
    auto newDir = pathLength2O >= 0 ? PropagationDirection::alongMomentum : PropagationDirection::oppositeToMomentum;

    if (currentPropDir == PropagationDirection::anyDirection) {
      currentPropDir = newDir;
    } else {
      if (newDir != currentPropDir) {
        solExists = false;
        return;
      }
    }

    //
    // Prepare iterations: count and total pathlength
    //

    pathLength = pathLength2O;
    auto iteration = maxIterations;

    while (notAtSurface(acc, plane, xnew, safeMaxDist)) {
      if (--iteration == 0) {
        //LogDebug("HelixArbitraryPlaneCrossing") << "pathLength : no convergence";
        solExists = false;
        return;
      }

      Vec3d pnew = directionInDouble(acc,
                                     pathLength,
                                     point,
                                     curvature,
                                     cosPhi0,
                                     sinPhi0,
                                     cosTheta,
                                     sinTheta,
                                     sinThetaI,
                                     theCachedS,
                                     theCachedDPhi,
                                     theCachedSDPhi,
                                     theCachedCDPhi);

      double tmpPathLength = 0.;
      bool tmpValidPath = false;
      //
      Vec3d tmpPosition(0.);
      Vec3d tmpDirectionOut(0.);

      // Originally it passes the theSinTheta
      helixArbitraryPlaneCrossing2Order<TAcc, propagators::PropagationDirection::anyDirection>(
          acc, xnew, pnew, curvature, plane, tmpPathLength, tmpValidPath, tmpPosition, tmpDirectionOut);
      /////////////////////////

      if (!tmpValidPath) {
        solExists = false;
        return;
      }

      pathLength += tmpPathLength;

      newDir = pathLength >= 0 ? PropagationDirection::alongMomentum : PropagationDirection::oppositeToMomentum;
      if (currentPropDir == PropagationDirection::anyDirection) {
        currentPropDir = newDir;
      } else {
        if (newDir != currentPropDir) {
          solExists = false;
          return;
        }
      }
      xnew = positionInDouble(acc,
                              pathLength,
                              point,
                              curvature,
                              cosPhi0,
                              sinPhi0,
                              cosTheta,
                              sinTheta,
                              sinThetaI,
                              theCachedS,
                              theCachedDPhi,
                              theCachedSDPhi,
                              theCachedCDPhi);
    }

    solExists = true;
    position = xnew;
    dir = directionInDouble(acc,
                            pathLength,
                            point,
                            curvature,
                            cosPhi0,
                            sinPhi0,
                            cosTheta,
                            sinTheta,
                            sinThetaI,
                            theCachedS,
                            theCachedDPhi,
                            theCachedSDPhi,
                            theCachedCDPhi);
  }

}  // namespace propagators

#endif  // RecoEgamma_EgammaElectronAlgos_interface_helixArbitraryPlaneCrossing_h
