#ifndef HeterogeneousCore_AlpakaInterface_interface_random_h
#define HeterogeneousCore_AlpakaInterface_interface_random_h

#include <array>
#include <numbers>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/Philox.h"

namespace cms::alpakatools {

  // Four standard normal values from four random words, with the Box-Muller method.
  // The values use the math functions of the backend, so CPU and GPU results can
  // differ in the last bits.
  template <typename TAcc>
  ALPAKA_FN_ACC std::array<float, 4> normalFloat4(TAcc const& acc, edm::philox::Counter const& bits) {
    std::array<float, 4> values;
    for (unsigned int pair = 0; pair < 2; ++pair) {
      // 1 - u is in (0, 1], so the log is finite.
      float const u1 = 1.f - edm::philox::uniformFloat(bits[2 * pair]);
      float const u2 = edm::philox::uniformFloat(bits[2 * pair + 1]);
      float const radius = alpaka::math::sqrt(acc, -2.f * alpaka::math::log(acc, u1));
      float const angle = 2.f * std::numbers::pi_v<float> * u2;
      values[2 * pair] = radius * alpaka::math::cos(acc, angle);
      values[2 * pair + 1] = radius * alpaka::math::sin(acc, angle);
    }
    return values;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_random_h
