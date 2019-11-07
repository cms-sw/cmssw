#ifndef HeterogeneousCore_CUDACore_CUDAContextState_h
#define HeterogeneousCore_CUDACore_CUDAContextState_h

#include "HeterogeneousCore/CUDAUtilities/interface/SharedStreamPtr.h"

#include <memory>

/**
 * The purpose of this class is to deliver the device and CUDA stream
 * information from ExternalWork's acquire() to producer() via a
 * member/StreamCache variable.
 */
class CUDAContextState {
public:
  CUDAContextState() = default;
  ~CUDAContextState() = default;

  CUDAContextState(const CUDAContextState&) = delete;
  CUDAContextState& operator=(const CUDAContextState&) = delete;
  CUDAContextState(CUDAContextState&&) = delete;
  CUDAContextState& operator=(CUDAContextState&& other) = delete;

private:
  friend class CUDAScopedContextAcquire;
  friend class CUDAScopedContextProduce;
  friend class CUDAScopedContextTask;

  void set(int device, cudautils::SharedStreamPtr stream) {
    throwIfStream();
    device_ = device;
    stream_ = std::move(stream);
  }

  int device() const { return device_; }

  const cudautils::SharedStreamPtr& streamPtr() const {
    throwIfNoStream();
    return stream_;
  }

  cudautils::SharedStreamPtr releaseStreamPtr() {
    throwIfNoStream();
    // This function needs to effectively reset stream_ (i.e. stream_
    // must be empty after this function). This behavior ensures that
    // the SharedStreamPtr is not hold for inadvertedly long (i.e. to
    // the next event), and is checked at run time.
    return std::move(stream_);
  }

  void throwIfStream() const;
  void throwIfNoStream() const;

  cudautils::SharedStreamPtr stream_;
  int device_;
};

#endif
