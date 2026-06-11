/**
 Description: Function to propagate from a point to a plane on the GPU
*/

#ifndef RecoEgamma_EgammaElectronAlgos_interface_helixBarrelPlaneCrossing_h
#define RecoEgamma_EgammaElectronAlgos_interface_helixBarrelPlaneCrossing_h

#include <cmath>

#include "RecoEgamma/EgammaElectronAlgos/interface/Plane.h"

#include "DataFormats/EgammaReco/interface/alpaka/Phys3DVector.h"

using Vec3d = cms::alpakatools::math::Phys3DVector<double>;

namespace propagators {

  enum class PropagationDirection { alongMomentum, oppositeToMomentum, anyDirection, invalidDirection };

  template <PropagationDirection propDir>
  constexpr Vec3d chooseSolution(const Vec3d& d1,
                                 const Vec3d& d2,
                                 const Vec3d& startingPos,
                                 const Vec3d& startingDir,
                                 int& theActualDir,
                                 bool& theSolExists) {
    Vec3d theD;

    const double momProj1 = startingDir[0] * d1[0] + startingDir[1] * d1[1];
    const double momProj2 = startingDir[0] * d2[0] + startingDir[1] * d2[1];

    const double d1_norm2 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
    const double d2_norm2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];

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
      constexpr double propSign = (propDir == PropagationDirection::alongMomentum) ? 1 : -1;
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

  template <typename TAcc, PropagationDirection propDir>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixBarrelPlaneCrossing(TAcc const& acc,
                                                                    const Vec3d& startingPos,
                                                                    const Vec3d& startingDir,
                                                                    const double rho,
                                                                    Vec3d& surfPosition,
                                                                    Vec3d& surfRotation,
                                                                    bool& theSolExists,
                                                                    Vec3d& position,
                                                                    Vec3d& direction,
                                                                    double& s) {
    const egamma::Plane<typename Vec3d::value_type> plane(surfPosition, surfRotation);

    constexpr double straightLineCutoff = 1.e-7;

    const double abs_rho = alpaka::math::abs(acc, rho);
    const double startingDir_2dnorm = startingPos.partial_norm(acc);

    auto compute_position = [&](const double s) -> Vec3d {
      const double norm = startingDir.norm(acc);
      const double scale = norm > 0. ? s / norm : 0.;  //that is, for "zero" vector this will be identity operation
      return cms::alpakatools::math::axpy(scale, startingDir, startingPos);
    };

    if (abs_rho < straightLineCutoff && abs_rho * startingDir_2dnorm < straightLineCutoff) {
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

    const double pt = startingDir.partial_norm(acc);

    const double o = 1. / (pt * rho);
    const double theXCenter = startingPos[0] - startingDir[1] * o;
    const double theYCenter = startingPos[1] + startingDir[0] * o;

    // This is default when there curvature is non zero
    const Vec3d n = plane.normalVector();

    const double distToPlane = -plane.localZ(startingPos);

    const double nx = n[0];
    const double ny = n[1];

    const double distCx = startingPos[0] - theXCenter;
    const double distCy = startingPos[1] - theYCenter;

    double nfac, dfac;
    double A, B, C;
    bool solveForX;

    if (alpaka::math::abs(acc, nx) > alpaka::math::abs(acc, ny)) {
      solveForX = false;
      nfac = ny / nx;
      dfac = distToPlane / nx;
      B = distCy - nfac * distCx;  // only part of B
      C = (2. * distCx + dfac) * dfac;
    } else {
      solveForX = true;
      nfac = nx / ny;
      dfac = distToPlane / ny;
      B = distCx - nfac * distCy;  // only part of B
      C = (2. * distCy + dfac) * dfac;
    }

    B -= nfac * dfac;
    B *= 2;  // the rest of B
    A = 1. + nfac * nfac;

    // Check solution existence first:
    const double D = B * B - 4 * A * C;

    if (D < 0) {
      theSolExists = false;
      return;
    }

    const double Q = (-0.5 * (B + alpaka::math::copysign(acc, alpaka::math::sqrt(acc, D), B)));

    const double first = Q / A;
    const double second = C / Q;

    Vec3d d1, d2;

    if (solveForX) {
      d1 = Vec3d(first, dfac - nfac * first, 0.0);
      d2 = Vec3d(second, dfac - nfac * second, 0.0);
    } else {
      d1 = Vec3d(dfac - nfac * first, first, 0.0);
      d2 = Vec3d(dfac - nfac * second, second, 0.0);
    }

    Vec3d theD;

    int theActualDir;

    theD = chooseSolution<propDir>(d1, d2, startingPos, startingDir, theActualDir, theSolExists);

    if (!theSolExists)
      return;

    const double scaled_dMag_rho = 0.5 * theD.norm(acc) * rho;  // theD.norm()

    double sinAlpha = scaled_dMag_rho;

    const double ipabs = 1. / startingDir.norm(acc);

    const double sinTheta = pt * ipabs;
    const double cosTheta = startingDir[2] * ipabs;

    if (alpaka::math::abs(acc, sinAlpha) > 1.0)
      sinAlpha = alpaka::math::copysign(acc, 1.0, sinAlpha);

    // Path length
    s = theActualDir * 2.0 * alpaka::math::asin(acc, sinAlpha) / (rho * sinTheta);

    // Position
    position = Vec3d(startingPos[0] + theD[0], startingPos[1] + theD[1], startingPos[2] + s * cosTheta);

    // Direction
    const double tmp = s >= 0 ? scaled_dMag_rho : -scaled_dMag_rho;
    const double tmp2 = tmp * tmp;

    const double sinPhi = (1. < tmp2) ? 0. : 2.0 * tmp * alpaka::math::sqrt(acc, 1. - tmp2);
    const double cosPhi = 1.0 - 2.0 * tmp2;

    direction = Vec3d(startingDir[0] * cosPhi - startingDir[1] * sinPhi,
                      startingDir[0] * sinPhi + startingDir[1] * cosPhi,
                      startingDir[2]);
  }

}  // namespace propagators

#endif  // RecoEgamma_EgammaElectronAlgos_interface_helixBarrelPlaneCrossing_h
