#ifndef DataFormats_Math_constexpr_cmath_h
#define DataFormats_Math_constexpr_cmath_h

#include <cstdint>

namespace reco {
  constexpr int32_t ceil(float num) {
    return (static_cast<float>(static_cast<int32_t>(num)) == num) ? static_cast<int32_t>(num)
                                                                  : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
  }
};  // namespace reco
#endif
