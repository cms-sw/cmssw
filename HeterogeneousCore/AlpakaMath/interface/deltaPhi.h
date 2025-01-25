#ifndef HeterogeneousCore_AlpakaMath_deltaPhi_h
#define HeterogeneousCore_AlpakaMath_deltaPhi_h

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  // reduce to [-pi,pi]
  template <typename TAcc, std::floating_point T>
  ALPAKA_FN_HOST_ACC inline T reducePhiRange(TAcc const& acc, T x) {
    constexpr T o2pi = T{1.} / (T{2.} * std::numbers::pi_v<T>);
    if (alpaka::math::abs(acc, x) <= std::numbers::pi_v<T>)
      return x;
    T n = alpaka::math::round(acc, x * o2pi);
    return x - n * T{2.} * std::numbers::pi_v<T>;
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_HOST_ACC inline T phi(TAcc const& acc, T x, T y) {
    return reducePhiRange(acc, std::numbers::pi_v<T> + alpaka::math::atan2(acc, -y, -x));
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_HOST_ACC inline T deltaPhi(TAcc const& acc, T x1, T y1, T x2, T y2) {
    return reducePhiRange(acc, alpaka::math::atan2(acc, -y2, -x2) - alpaka::math::atan2(acc, -y1, -x1));
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_HOST_ACC inline T deltaPhi(TAcc const& acc, T phi1, T phi2) {
    return reducePhiRange(acc, phi1 - phi2);
  }

}  // namespace cms::alpakatools

#endif
