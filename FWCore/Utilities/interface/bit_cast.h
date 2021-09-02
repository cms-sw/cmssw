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

// system include files
#include <cstring>

// user include files

namespace edm {
  //in C++20 we can use std::bit_cast which is constexpr
  template <class To, class From>
  inline To bit_cast(const From &src) noexcept {
    static_assert(sizeof(To) == sizeof(From), "incompatible types");
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
  }
}  // namespace edm
#endif
