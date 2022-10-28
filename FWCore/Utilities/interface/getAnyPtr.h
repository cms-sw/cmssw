#ifndef FWCore_Utilities_getAnyPtr_h
#define FWCore_Utilities_getAnyPtr_h

#include <cassert>
#include <memory>

namespace edm {
  template <typename T>
  inline std::unique_ptr<T> getAnyPtr(void *p, int offset) {
    return std::unique_ptr<T>(static_cast<T *>(static_cast<void *>(static_cast<unsigned char *>(p) + offset)));
  }
}  // namespace edm

#endif
