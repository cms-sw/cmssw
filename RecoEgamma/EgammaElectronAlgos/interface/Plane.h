#ifndef RecoEgamma_EgammaElectronAlgos_interface_Plane_h
#define RecoEgamma_EgammaElectronAlgos_interface_Plane_h

#include <cmath>
#include "DataFormats/EgammaReco/interface/alpaka/Phys3DVector.h"

namespace egamma {

  template <typename T = double>
  class Plane {
  public:
    using Vec3 = cms::alpakatools::math::Phys3DVector<T>;

    // Constructor
    ALPAKA_FN_HOST_ACC Plane(const Vec3& pos, const Vec3& rot) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < 3; i++) {
        position[i] = pos[i];
        rotation[i] = rot[i];
      }
    }
    // Returns the position of the plane
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vec3 pos() const { return Vec3(position[0], position[1], position[2]); }

    // Returns a specific component of the position of the plane
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T pos(const unsigned int x) const { return position[x]; }

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T pos_norm(TAcc const& acc) const {
      return position.norm(acc);
    }

    // Returns the normal vector of the plane
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Vec3 normalVector() const {
      return Vec3(rotation[0], rotation[1], rotation[2]);
    }

    // Fast access to distance from plane for a point
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T localZ(const Vec3& vp) const {
      T tmp_dot{0};
      CMS_UNROLL_LOOP
      for (unsigned int i = 0; i < 3; i++) {
        tmp_dot += rotation[i] * (vp[i] - position[i]);
      }
      return tmp_dot;
    }

    // Clamped distance from plane for a point
    template <typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T localZclamped(TAcc const& acc, const Vec3& vp) const {
      const T d = localZ(vp);
      return alpaka::math::abs(acc, d) > 1e-7f ? d : 0;
    }

    // Fast access to distance from plane for a vector
    template <typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T distanceFromPlaneVector(TAcc const& acc, const Vec3& gv) const {
      return cms::alpakatools::math::dot(rotation, gv);
    }

  private:
    Vec3 position;
    Vec3 rotation;  // z coordinate of rotation matrix
  };
}  // namespace egamma

#endif  // RecoEgamma_EgammaElectronAlgos_interface_Plane_h
