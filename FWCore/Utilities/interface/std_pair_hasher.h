#ifndef FWCore_Utilities_std_pair_hasher_h
#define FWCore_Utilities_std_pair_hasher_h
/*
 tbb::hash was changed to used std::hash which does not have an implementation for std::pair.
 This hasher is taken from the boost::hash implementation.
*/
#include "FWCore/Utilities/interface/zero_allocator.h"

namespace edm {
  struct StdPairHasher {
    std::size_t operator()(const std::pair<const std::string, const std::string>& a) const noexcept {
      return edm::hash_value(a.first, a.second);
    }
    std::size_t operator()(const std::pair<const unsigned int, const unsigned int>& a) const noexcept {
      return edm::hash_value(a.first, a.second);
    }
  };
}  // namespace edm
#endif
