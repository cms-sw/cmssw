#ifndef DataFormats_EgammaReco_interface_EleRelPointPairPortable_h
#define DataFormats_EgammaReco_interface_EleRelPointPairPortable_h

#include <cmath>
#include <numbers>
#include "DataFormats/EgammaReco/interface/alpaka/Phys3DVector.h"

//==========================================================================
// When wanting to compute and compare several characteristics of one or two
// points, relatively to a given origin, using GPU-friendly Phys3DVector
//============================================================================

namespace egamma {
  using namespace cms::alpakatools;

  // Helper function to compute relative position
  template <typename T>
  constexpr auto relativePosition(const math::Phys3DVector<T>& point, const math::Phys3DVector<T>& origin)
      -> math::Phys3DVector<T> {
    return math::xmy(point, origin);
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto relative_eta(TAcc const& acc,
                                                        const math::Phys3DVector<T>& p,
                                                        const math::Phys3DVector<T>& origin) -> T {
    const T tmp = math::diff_norm2(p, origin);
    const T pdiff = alpaka::math::sqrt(acc, tmp);
    const T z = p[2] - origin[2];

    return static_cast<T>(0.5) * alpaka::math::log(acc, (pdiff + z) / (pdiff - z));
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto reduceRange(TAcc const& acc, const T x) -> T {
    constexpr T o2pi = static_cast<T>(0.5) * std::numbers::inv_pi;
    if (alpaka::math::abs(acc, x) <= std::numbers::pi)
      return x;
    return x - alpaka::math::floor(acc, x * o2pi + (x < 0 ? -static_cast<T>(0.5) : static_cast<T>(0.5))) *
                   static_cast<T>(2) * std::numbers::pi;
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto relative_phi(TAcc const& acc,
                                                        const math::Phys3DVector<T>& p1,
                                                        const math::Phys3DVector<T>& p2) -> T {
    const T phi = alpaka::math::atan2(acc, p1[1], p1[0]) - alpaka::math::atan2(acc, p2[1], p2[0]);
    return reduceRange(acc, phi);
  }

  template <typename T = double>
  class EleRelPointPairPortable {
  public:
    using Vec3 = cms::alpakatools::math::Phys3DVector<T>;

    // Constructor to compute relative points
    constexpr EleRelPointPairPortable(const Vec3& p1, const Vec3& p2, const Vec3& origin)
        : relP1(relativePosition(p1, origin)), relP2(relativePosition(p2, origin)) {}

    // Calculate differences
    //constexpr auto dEta() const { return relative_eta(relP1, relP2); }
    constexpr inline T dZ() const { return (relP1[2] - relP2[2]); }

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T dPerp(TAcc const& acc) const {
      const T relP1_rho = relP1.rho(acc);
      const T relP2_rho = relP2.rho(acc);

      return (relP1_rho - relP2_rho);
    }

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto dPhi(TAcc const& acc) const {
      return relative_phi(acc, relP1, relP2);
    }

  private:
    Vec3 relP1;  // Relative point 1
    Vec3 relP2;  // Relative point 2
  };

}  // namespace egamma

#endif  // DataFormats_EgammaReco_interface_EleRelPointPairPortable_h
