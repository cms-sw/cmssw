#ifndef L1Trigger_L1TGEM_ME0StubFit_H
#define L1Trigger_L1TGEM_ME0StubFit_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <array>
#include <ap_int.h>
#include <ap_fixed.h>
#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"

namespace l1t {
  namespace me0 {
    inline constexpr std::array<double, 2047> RECIP = [] {
      std::array<double, 2047> lut{};
      for (int i = 1; i <= 2047; ++i) {
        lut[i - 1] = 1.0 / static_cast<double>(i);
      }
      return lut;
    }();

    double reciprocal(int n);
    double reciprocal6(int n);
    std::vector<double> llseFit(const std::vector<double>& x, const std::vector<double>& y);
    std::vector<double> vhdlExactFit(const std::vector<int>& centroids,
                                     const std::vector<bool>& validMask,
                                     bool verbose = false);

  }  // namespace me0
}  // namespace l1t

#endif
