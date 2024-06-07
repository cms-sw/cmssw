#ifndef FWCore_Utilities_bit_cast_h
#define FWCore_Utilities_bit_cast_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     bit_cast
//
/**\function edm::bit_cast bit_cast.h "FWCore/Utilities/interface/bit_cast.h"

 Description: C++ 20 std::bit_cast stand-in

 Usage:
    See documentation on std::bit_cast in C++ 20

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 01 Sep 2021 19:11:41 GMT
//

// for compilers that do not support __has_builtin
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

// system include files
#include <cstring>
#include <type_traits>

#if __cplusplus >= 202002L

// in C++20 we can use std::bit_cast

#include <bit>

namespace edm {
  using std::bit_cast;
}  // namespace edm

#elif __has_builtin(__builtin_bit_cast)

// before C++20 we can use __builtin_bit_cast, if supported

namespace edm {
  template <typename To, typename From>
  constexpr inline To bit_cast(const From &src) noexcept {
    static_assert(std::is_trivially_copyable_v<From>);
    static_assert(std::is_trivially_copyable_v<To>);
    static_assert(sizeof(To) == sizeof(From), "incompatible types");
    return __builtin_bit_cast(To, src);
  }
}  // namespace edm

#else

#error constexpr edm::bit_cast is not supported by the compiler

#endif  // __cplusplus >= 202002L

#endif  // FWCore_Utilities_bit_cast_h
