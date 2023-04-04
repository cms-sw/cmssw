#ifndef AlpakaCore_alpakastdAlgorithm_h
#define AlpakaCore_alpakastdAlgorithm_h

#include <algorithm>
#include <functional>
#include <utility>

#include <alpaka/alpaka.hpp>

// reimplementation of std algorithms able to compile with Alpaka,
// mostly by declaring them constexpr

namespace alpaka_std {

  template <typename RandomIt, typename T, typename Compare = std::less<T>>
  ALPAKA_FN_HOST_ACC constexpr RandomIt upper_bound(RandomIt first, RandomIt last, const T &value, Compare comp = {}) {
    auto count = last - first;

    while (count > 0) {
      auto it = first;
      auto step = count / 2;
      it += step;
      if (!comp(value, *it)) {
        first = ++it;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    return first;
  }

}  // namespace alpaka_std

#endif  // AlpakaCore_alpakastdAlgorithm_h
