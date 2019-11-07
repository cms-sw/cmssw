#ifndef CUDADataFormats_Common_CUDAProductBase_h
#define CUDADataFormats_Common_CUDAProductBase_h

#include <atomic>
#include <memory>

#include <cuda/api_wrappers.h>

#include "HeterogeneousCore/CUDAUtilities/interface/SharedStreamPtr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedEventPtr.h"

namespace impl {
  class CUDAScopedContextBase;
}

/**
 * Base class for all instantiations of CUDA<T> to hold the
 * non-T-dependent members.
 */
class CUDAProductBase {
public:
  CUDAProductBase() = default;  // Needed only for ROOT dictionary generation
  ~CUDAProductBase();

  CUDAProductBase(const CUDAProductBase&) = delete;
  CUDAProductBase& operator=(const CUDAProductBase&) = delete;
  CUDAProductBase(CUDAProductBase&& other)
      : stream_{std::move(other.stream_)},
        event_{std::move(other.event_)},
        mayReuseStream_{other.mayReuseStream_.load()},
        device_{other.device_} {}
  CUDAProductBase& operator=(CUDAProductBase&& other) {
    stream_ = std::move(other.stream_);
    event_ = std::move(other.event_);
    mayReuseStream_ = other.mayReuseStream_.load();
    device_ = other.device_;
    return *this;
  }

  bool isValid() const { return stream_.get() != nullptr; }
  bool isAvailable() const;

  int device() const { return device_; }

  // cudaStream_t is a pointer to a thread-safe object, for which a
  // mutable access is needed even if the CUDAScopedContext itself
  // would be const. Therefore it is ok to return a non-const
  // pointer from a const method here.
  cudaStream_t stream() const { return stream_.get(); }

  // cudaEvent_t is a pointer to a thread-safe object, for which a
  // mutable access is needed even if the CUDAScopedContext itself
  // would be const. Therefore it is ok to return a non-const
  // pointer from a const method here.
  cudaEvent_t event() const { return event_ ? event_.get() : nullptr; }

protected:
  explicit CUDAProductBase(int device, cudautils::SharedStreamPtr stream)
      : stream_{std::move(stream)}, device_{device} {}

private:
  friend class impl::CUDAScopedContextBase;
  friend class CUDAScopedContextProduce;

  // The following functions are intended to be used only from CUDAScopedContext
  void setEvent(cudautils::SharedEventPtr event) { event_ = std::move(event); }
  const cudautils::SharedStreamPtr& streamPtr() const { return stream_; }

  bool mayReuseStream() const {
    bool expected = true;
    bool changed = mayReuseStream_.compare_exchange_strong(expected, false);
    // If the current thread is the one flipping the flag, it may
    // reuse the stream.
    return changed;
  }

  // The cudaStream_t is really shared among edm::Event products, so
  // using shared_ptr also here
  cudautils::SharedStreamPtr stream_;  //!
  // shared_ptr because of caching in CUDAEventCache
  cudautils::SharedEventPtr event_;  //!

  // This flag tells whether the CUDA stream may be reused by a
  // consumer or not. The goal is to have a "chain" of modules to
  // queue their work to the same stream.
  mutable std::atomic<bool> mayReuseStream_ = true;  //!

  // The CUDA device associated with this product
  int device_ = -1;  //!
};

#endif
