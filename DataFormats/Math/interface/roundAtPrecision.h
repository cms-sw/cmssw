#ifndef DataFormatsMathroundAtPrecision_H
#define DataFormatsMathroundAtPrecision_H

#include "FWCore/Utilities/interface/isFinite.h"
#include <cstring>
#include <cstdint>

// round with N significant bits
template <int N>
float roundAtPrecision(float x) {
  if (edm::isNotFinite(x))
    return x;

  static_assert(N < 23);
  constexpr auto shift = 23 - N;
  constexpr uint32_t mask = 1 << (shift - 1);

  uint32_t i;
  memcpy(&i, &x, sizeof(x));

  i += mask;
  i >>= shift;
  i <<= shift;
  memcpy(&x, &i, sizeof(x));

  return x;
}

template <>
float roundAtPrecision<23>(float x) {
  return x;
}

#endif  // DataFormatsMathroundAtPrecision_H
