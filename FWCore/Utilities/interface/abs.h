#ifndef FWCore_Utilities_Interface_abs_h
#define FWCore_Utilities_Interface_abs_h

#include <concepts>

namespace edm {

  template <class T>
  concept arithmetic = std::integral<T> || std::floating_point<T>;

  template <arithmetic T>
  constexpr T abs(T x) noexcept {
    if constexpr (std::unsigned_integral<T>) {
      return x;
    } else {
      return x < T{0} ? T{-x} : x;
    }
  }

}  // namespace edm

#endif  //FWCore_Utilities_Interface_abs_h
