#ifndef HeterogeneousCore_CUDACore_CUDAContextState_h
#define HeterogeneousCore_CUDACore_CUDAContextState_h

#include <cuda/api_wrappers.h>

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

  void set(int device, std::shared_ptr<cuda::stream_t<>> stream) {
    throwIfStream();
    device_ = device;
    stream_ = std::move(stream);
  }

  int device() const { return device_; }

  std::shared_ptr<cuda::stream_t<>>& streamPtr() {
    throwIfNoStream();
    return stream_;
  }

  const std::shared_ptr<cuda::stream_t<>>& streamPtr() const {
    throwIfNoStream();
    return stream_;
  }

  void throwIfStream() const;
  void throwIfNoStream() const;

  std::shared_ptr<cuda::stream_t<>> stream_;
  int device_;
};

#endif
