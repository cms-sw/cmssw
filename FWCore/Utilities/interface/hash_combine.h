#ifndef FWCore_Utilities_hash_combine_h
#define FWCore_Utilities_hash_combine_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     hash_combine
//
/**\function hash_combine hash_combine.h "FWCore/Utilities/interface/hash_combine.h"

 Description: Convenience Functions to Combine Hash Values

 Based on http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3876.pdf
 Combination algorithm is the same as boost::hash_combine

 Usage:
    <usage>

*/

// system include files
#include <functional>

namespace edm {
  template <typename T>
  inline void hash_combine(std::size_t& seed, const T& val) {
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  template <typename T, typename... Types>
  inline void hash_combine(std::size_t& seed, const T& val, const Types&... args) {
    hash_combine(seed, val);
    hash_combine(seed, args...);
  }

  template <typename... Types>
  inline std::size_t hash_value(const Types&... args) {
    std::size_t seed{0};
    hash_combine(seed, args...);
    return seed;
  }
}  // namespace edm

#endif
