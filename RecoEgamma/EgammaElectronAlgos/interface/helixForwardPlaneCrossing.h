/**
 Description: Function to propagate a helix from a point to a plane
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_helixForwardPlaneCrossing_h
#define RecoEgamma_EgammaElectronAlgos_interface_helixForwardPlaneCrossing_h

#include <cfloat>
#include <cmath>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "RecoEgamma/EgammaElectronAlgos/interface/Phys3DVector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/Plane.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/helixBarrelPlaneCrossingByCircle.h"

namespace propagators {

  // ---------------------------------------------------------
  //  Position of helix after path-length s
  // ---------------------------------------------------------
  template <typename T>
  constexpr Vec3<T> positionInHelix(const bool select,
                                    const T s,
                                    const Vec3<T>& point,
                                    const T rho,
                                    const T cosPhi0,
                                    const T sinPhi0,
                                    const T cosTheta,
                                    const T sinTheta,
                                    const T cachedS,
                                    const T cachedSDPhi,
                                    const T cachedCDPhi) {
    if (select) {
      const T o = static_cast<T>(1.) / rho;
      return Vec3<T>(point[0] + (-sinPhi0 * (static_cast<T>(1.) - cachedCDPhi) + cosPhi0 * cachedSDPhi) * o,
                     point[1] + (cosPhi0 * (static_cast<T>(1.) - cachedCDPhi) + sinPhi0 * cachedSDPhi) * o,
                     point[2] + s * cosTheta);
    } else {
      const T st = cachedS * sinTheta;
      return Vec3<T>(point[0] + (cosPhi0 - st * static_cast<T>(0.5) * rho * sinPhi0) * st,
                     point[1] + (sinPhi0 + st * static_cast<T>(0.5) * rho * cosPhi0) * st,
                     point[2] + st * cosTheta / sinTheta);
    }
  }

  // ---------------------------------------------------------
  //  Direction of helix after path-length s
  // ---------------------------------------------------------
  template <typename T>
  constexpr Vec3<T> directionInHelix(const bool select,
                                     const T s,
                                     const T rho,
                                     const T cosPhi0,
                                     const T sinPhi0,
                                     const T cosTheta,
                                     const T sinTheta,
                                     const T cachedSDPhi,
                                     const T cachedCDPhi) {
    if (select) {
      return Vec3<T>(cosPhi0 * cachedCDPhi - sinPhi0 * cachedSDPhi,
                     sinPhi0 * cachedCDPhi + cosPhi0 * cachedSDPhi,
                     cosTheta / sinTheta);
    } else {
      const T dph = s * rho * sinTheta;
      return Vec3<T>(cosPhi0 - (sinPhi0 + static_cast<T>(0.5) * cosPhi0 * dph) * dph,
                     sinPhi0 + (cosPhi0 - static_cast<T>(0.5) * sinPhi0 * dph) * dph,
                     cosTheta / sinTheta);
    }
  }

  // ---------------------------------------------------------
  //  Main propagation function
  // ---------------------------------------------------------
  template <typename TAcc, PropagationDirection propDir, typename T>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixForwardPlaneCrossing(TAcc const& acc,
                                                                     const Vec3<T>& point,
                                                                     const Vec3<T>& direction,
                                                                     const float curvature,
                                                                     const egamma::Plane<T> plane,
                                                                     T& pathLength,
                                                                     Vec3<T>& position,
                                                                     Vec3<T>& dir,
                                                                     bool& solExists) {
    T cachedS = static_cast<T>(0.);
    T cachedDPhi = static_cast<T>(0.);
    T cachedSDPhi = static_cast<T>(0.);
    T cachedCDPhi = static_cast<T>(1.);

    const T px = direction[0];
    const T py = direction[1];
    const T pz = direction[2];

    const T pt2 = px * px + py * py;
    const T p2 = pt2 + pz * pz;

    const T pI = static_cast<T>(1.) / alpaka::math::sqrt(acc, p2);
    const T ptI = static_cast<T>(1.) / alpaka::math::sqrt(acc, pt2);

    const T cosPhi0 = px * ptI;
    const T sinPhi0 = py * ptI;

    const T cosTheta = pz * pI;
    const T sinTheta = pt2 * ptI * pI;

    // ---------------------------------------------------------
    //  Path length to plane
    // ---------------------------------------------------------
    const bool min_cosTheta_flag = (alpaka::math::abs(acc, cosTheta) < std::numeric_limits<float>::min());

    pathLength = min_cosTheta_flag ? static_cast<T>(0.) : (plane.pos(2) - point[2]) / cosTheta;

    const bool validSolution =
        !min_cosTheta_flag && !(((propDir == PropagationDirection::alongMomentum) && (pathLength < 0.)) ||
                                ((propDir == PropagationDirection::oppositeToMomentum) && (pathLength > 0.)) ||
                                !alpaka::math::isfinite(acc, pathLength));

    if (!validSolution) {
      solExists = false;
      pathLength = static_cast<T>(0.);
      return;
    }

    // ---------------------------------------------------------
    //  Update cached helix terms
    // ---------------------------------------------------------
    if (pathLength != cachedS) {
      cachedS = pathLength;
      cachedDPhi = cachedS * curvature * sinTheta;
      cachedSDPhi = alpaka::math::sin(acc, cachedDPhi);
      cachedCDPhi = alpaka::math::cos(acc, cachedDPhi);
    }

    const bool cachedDPhi_flag = alpaka::math::abs(acc, cachedDPhi) > static_cast<T>(1.e-4);

    // ---------------------------------------------------------
    //  Compute final position and direction
    // ---------------------------------------------------------
    position = positionInHelix<T>(cachedDPhi_flag,
                                  pathLength,
                                  point,
                                  curvature,
                                  cosPhi0,
                                  sinPhi0,
                                  cosTheta,
                                  sinTheta,
                                  cachedS,
                                  cachedSDPhi,
                                  cachedCDPhi);

    dir = directionInHelix<T>(
        cachedDPhi_flag, pathLength, curvature, cosPhi0, sinPhi0, cosTheta, sinTheta, cachedSDPhi, cachedCDPhi);

    solExists = true;
  }

}  // namespace propagators

#endif  // RecoEgamma_EgammaElectronAlgos_interface_helixForwardPlaneCrossing_h
