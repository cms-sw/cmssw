#ifndef RecoTracker_MkFitCore_src_Matriplex_Memory_h
#define RecoTracker_MkFitCore_src_Matriplex_Memory_h

#include <cstdlib>

namespace Matriplex {

  constexpr std::size_t round_up_align64(std::size_t size) {
    constexpr std::size_t mask = 64 - 1;
    return size & mask ? (size & ~mask) + 64 : size;
  }

  inline void* aligned_alloc64(std::size_t size) { return std::aligned_alloc(64, round_up_align64(size)); }

}  // namespace Matriplex

#endif
