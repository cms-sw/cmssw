#ifndef FWCore_Framework_CacheHandle_h
#define FWCore_Framework_CacheHandle_h

/** \class edm::CacheHandle

\author W. David Dagenhart, created 25 March, 2021

*/

#include "FWCore/Utilities/interface/Exception.h"

namespace edm {

  template <typename T>
  class CacheHandle {
  public:
    CacheHandle() : data_(nullptr) {}
    CacheHandle(T const* data) : data_(data) {}

    T const* get() const {
      if (!isValid()) {
        throw cms::Exception("InvalidCache") << "CacheHandle is invalid";
      }
      return data_;
    }
    T const* operator->() const { return get(); }
    T const& operator*() const { return *get(); }

    bool isValid() const { return data_ != nullptr; }

  private:
    T const* data_;
  };
}  // namespace edm
#endif
