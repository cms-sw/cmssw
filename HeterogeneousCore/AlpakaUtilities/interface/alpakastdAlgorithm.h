#ifndef AlpakaCore_alpakastdAlgorithm_h
#define AlpakaCore_alpakastdAlgorithm_h

#include <algorithm>
#include <functional>
#include <utility>

#include <alpaka/alpaka.hpp>

// reimplementation of std algorithms able to compile with Alpaka,
// mostly by declaring them constexpr

namespace alpaka_std {

  template <typename T = void>
  struct less {
    ALPAKA_FN_HOST_ACC constexpr bool operator()(const T &lhs, const T &rhs) const { return lhs < rhs; }
  };

  template <>
  struct less<void> {
    template <typename T, typename U>
    ALPAKA_FN_HOST_ACC constexpr bool operator()(const T &lhs, const U &rhs) const {
      return lhs < rhs;
    }
  };

  template <typename RandomIt, typename T, typename Compare = less<T>>
  ALPAKA_FN_HOST_ACC constexpr RandomIt lower_bound(RandomIt first, RandomIt last, const T &value, Compare comp = {}) {
    auto count = last - first;

    while (count > 0) {
      auto it = first;
      auto step = count / 2;
      it += step;
      if (comp(*it, value)) {
        first = ++it;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    return first;
  }

  template <typename RandomIt, typename T, typename Compare = less<T>>
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

  template <typename RandomIt, typename T, typename Compare = alpaka_std::less<T>>
  ALPAKA_FN_HOST_ACC constexpr RandomIt binary_find(RandomIt first, RandomIt last, const T &value, Compare comp = {}) {
    first = alpaka_std::lower_bound(first, last, value, comp);
    return first != last && !comp(value, *first) ? first : last;
  }

}  // namespace alpaka_std

#endif  // AlpakaCore_alpakastdAlgorithm_h
