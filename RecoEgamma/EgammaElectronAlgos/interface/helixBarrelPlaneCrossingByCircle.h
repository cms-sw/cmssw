/**
 Description: Function to propagate from a point to a plane on the GPU
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_helixBarrelPlaneCrossingByCircle_h
#define RecoEgamma_EgammaElectronAlgos_interface_helixBarrelPlaneCrossingByCircle_h

#include <cmath>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "RecoEgamma/EgammaElectronAlgos/interface/Phys3DVector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/Plane.h"

// Kept for the other propagator headers and their users, which still name the
// double precision instantiation explicitly.
using Vec3d = egamma::math::Phys3DVector<double>;

namespace propagators {

  // The floating point type is generic: instantiate with double or float.
  template <typename T>
  using Vec3 = egamma::math::Phys3DVector<T>;

  enum class PropagationDirection { alongMomentum, oppositeToMomentum, anyDirection, invalidDirection };

  template <PropagationDirection propDir, typename T>
  constexpr Vec3<T> chooseSolution(const Vec3<T>& d1,
                                   const Vec3<T>& d2,
                                   const Vec3<T>& startingPos,
                                   const Vec3<T>& startingDir,
                                   int& theActualDir,
                                   bool& theSolExists) {
    Vec3<T> theD;

    const T momProj1 = startingDir[0] * d1[0] + startingDir[1] * d1[1];
    const T momProj2 = startingDir[0] * d2[0] + startingDir[1] * d2[1];

    const T d1_norm2 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
    const T d2_norm2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];

    const bool selection_flag = d1_norm2 < d2_norm2;

    theSolExists = true;

    if constexpr (propDir == PropagationDirection::anyDirection) {
      if (selection_flag) {
        theD = d1;
        theActualDir = (momProj1 > 0) ? 1 : -1;
      } else {
        theD = d2;
        theActualDir = (momProj2 > 0) ? 1 : -1;
      }
    } else {
      constexpr T propSign = (propDir == PropagationDirection::alongMomentum) ? 1 : -1;
      if (momProj1 * momProj2 < 0) {
        theD = (momProj1 * propSign > 0) ? d1 : d2;
        theActualDir = propSign;
      } else if (momProj1 * propSign > 0) {
        theD = selection_flag ? d1 : d2;
        theActualDir = propSign;
      } else {
        theSolExists = false;
      }
    }

    return theD;
  }

  template <typename TAcc, PropagationDirection propDir, typename T>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixBarrelPlaneCrossing(TAcc const& acc,
                                                                    const Vec3<T>& startingPos,
                                                                    const Vec3<T>& startingDir,
                                                                    const T rho,
                                                                    Vec3<T>& surfPosition,
                                                                    Vec3<T>& surfRotation,
                                                                    bool& theSolExists,
                                                                    Vec3<T>& position,
                                                                    Vec3<T>& direction,
                                                                    T& s) {
    const egamma::Plane<T> plane(surfPosition, surfRotation);

    constexpr T straightLineCutoff = static_cast<T>(1.e-7);
    constexpr T eps = std::numeric_limits<T>::epsilon();

    const T abs_rho = alpaka::math::abs(acc, rho);
    const T startingDir_rho = startingPos.rho(acc);

    auto compute_position = [&](const T s) -> Vec3<T> {
      const T mag = startingDir.r(acc);
      const T scale =
          mag > eps ? s / mag : static_cast<T>(0.);  //that is, for "zero" vector this will be identity operation
      return egamma::math::axpy(acc, scale, startingDir, startingPos);
    };

    if (abs_rho < straightLineCutoff && abs_rho * startingDir_rho < straightLineCutoff) {
      // calculate path length
      const auto pz = plane.distanceFromPlaneVector(acc, startingDir);

      s = plane.localZclamped(acc, startingPos) / pz;

      if (s != 0) {
        theSolExists = true;
        position = compute_position(s);
        direction = startingDir;
      } else {
        theSolExists = false;
      }

      return;  // all needed data members have been set
    }

    const T pt = startingDir.rho(acc);

    const T o = static_cast<T>(1.) / (pt * rho);
    const T theXCenter = startingPos[0] - startingDir[1] * o;
    const T theYCenter = startingPos[1] + startingDir[0] * o;

    // This is default when there curvature is non zero
    const Vec3<T> n = plane.normalVector();

    const T distToPlane = -plane.localZ(startingPos);

    const T nx = n[0];
    const T ny = n[1];

    const T distCx = startingPos[0] - theXCenter;
    const T distCy = startingPos[1] - theYCenter;

    T nfac, dfac;
    T A, B, C;
    bool solveForX;

    if (alpaka::math::abs(acc, nx) > alpaka::math::abs(acc, ny)) {
      solveForX = false;
      nfac = ny / nx;
      dfac = distToPlane / nx;
      B = distCy - nfac * distCx;  // only part of B
      C = (static_cast<T>(2.) * distCx + dfac) * dfac;
    } else {
      solveForX = true;
      nfac = nx / ny;
      dfac = distToPlane / ny;
      B = distCx - nfac * distCy;  // only part of B
      C = (static_cast<T>(2.) * distCy + dfac) * dfac;
    }

    B -= nfac * dfac;
    B *= static_cast<T>(2);  // the rest of B
    A = static_cast<T>(1.) + nfac * nfac;

    // Check solution existence first:
    const T D = B * B - static_cast<T>(4) * A * C;

    if (D < 0) {
      theSolExists = false;
      return;
    }

    const T Q = (-static_cast<T>(0.5) * (B + alpaka::math::copysign(acc, alpaka::math::sqrt(acc, D), B)));

    const T first = Q / A;
    const T second = C / Q;

    Vec3<T> d1, d2;

    if (solveForX) {
      d1 = Vec3<T>(first, dfac - nfac * first, static_cast<T>(0.));
      d2 = Vec3<T>(second, dfac - nfac * second, static_cast<T>(0.));
    } else {
      d1 = Vec3<T>(dfac - nfac * first, first, static_cast<T>(0.));
      d2 = Vec3<T>(dfac - nfac * second, second, static_cast<T>(0.));
    }

    Vec3<T> theD;

    int theActualDir;

    theD = chooseSolution<propDir>(d1, d2, startingPos, startingDir, theActualDir, theSolExists);

    if (!theSolExists)
      return;

    const T scaled_dMag_rho = static_cast<T>(0.5) * theD.r(acc) * rho;  // theD.norm()

    T sinAlpha = scaled_dMag_rho;

    const T ipabs = static_cast<T>(1.) / startingDir.r(acc);

    const T sinTheta = pt * ipabs;
    const T cosTheta = startingDir[2] * ipabs;

    if (alpaka::math::abs(acc, sinAlpha) > static_cast<T>(1.))
      sinAlpha = alpaka::math::copysign(acc, static_cast<T>(1.), sinAlpha);

    // Path length
    s = theActualDir * static_cast<T>(2.) * alpaka::math::asin(acc, sinAlpha) / (rho * sinTheta);

    // Position
    position = Vec3<T>(startingPos[0] + theD[0], startingPos[1] + theD[1], startingPos[2] + s * cosTheta);

    // Direction
    const T tmp = s >= 0 ? scaled_dMag_rho : -scaled_dMag_rho;
    const T tmp2 = tmp * tmp;

    const T sinPhi = (static_cast<T>(1.) < tmp2)
                         ? static_cast<T>(0.)
                         : static_cast<T>(2.) * tmp * alpaka::math::sqrt(acc, static_cast<T>(1.) - tmp2);
    const T cosPhi = static_cast<T>(1.) - static_cast<T>(2.) * tmp2;

    direction = Vec3<T>(startingDir[0] * cosPhi - startingDir[1] * sinPhi,
                        startingDir[0] * sinPhi + startingDir[1] * cosPhi,
                        startingDir[2]);
  }

}  // namespace propagators

#endif  // RecoEgamma_EgammaElectronAlgos_interface_helixBarrelPlaneCrossingByCircle_h
