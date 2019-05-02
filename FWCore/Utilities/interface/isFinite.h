#ifndef FWCORE_Utilities_isFinite_H
#define FWCORE_Utilities_isFinite_H

namespace edm {
  template <typename T>
  constexpr bool isFinite(T x);

  template <typename T>
  constexpr bool isNotFinite(T x) {
    return !isFinite(x);
  }

  template <>
  constexpr bool isFinite(float x) {
    const unsigned int mask = 0x7f800000;
    union {
      unsigned int l;
      float d;
    } v = {.d = x};
    return (v.l & mask) != mask;
  }

  template <>
  constexpr bool isFinite(double x) {
    const unsigned long long mask = 0x7FF0000000000000LL;
    union {
      unsigned long long l;
      double d;
    } v = {.d = x};
    return (v.l & mask) != mask;
  }

  template <>
  constexpr bool isFinite(long double x) {
    double xx = x;
    return isFinite(xx);
  }

}  // namespace edm

#endif  // FWCORE_Utilities_isFinite_H
