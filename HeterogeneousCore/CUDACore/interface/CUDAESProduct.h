#ifndef HeterogeneousCore_CUDACore_CUDAESProduct_h
#define HeterogeneousCore_CUDACore_CUDAESProduct_h

#include <atomic>
#include <cassert>
#include <vector>

#include <cuda/api_wrappers.h>

#include "FWCore/Concurrency/interface/hardware_pause.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/CUDAUtilities/interface/eventIsOccurred.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAEventCache.h"

template <typename T>
class CUDAESProduct {
public:
  CUDAESProduct() : gpuDataPerDevice_(cuda::device::count()) {
    for (size_t i = 0; i < gpuDataPerDevice_.size(); ++i) {
      gpuDataPerDevice_[i].m_event = cudautils::getCUDAEventCache().getCUDAEvent();
    }
  }
  ~CUDAESProduct() = default;

  // transferAsync should be a function of (T&, cudaStream_t)
  // which enqueues asynchronous transfers (possibly kernels as well)
  // to the CUDA stream
  template <typename F>
  const T& dataForCurrentDeviceAsync(cudaStream_t cudaStream, F transferAsync) const {
    auto device = cuda::device::current::get().id();

    auto& data = gpuDataPerDevice_[device];

    // If GPU data has already been filled, we can return it
    // immediately
    if (not data.m_filled.load()) {
      // It wasn't, so need to fill it
      std::scoped_lock<std::mutex> lk{data.m_mutex};

      if (data.m_filled.load()) {
        // Other thread marked it filled while we were locking the mutex, so we're free to return it
        return data.m_data;
      }

      if (data.m_fillingStream != nullptr) {
        // Someone else is filling

        // Check first if the recorded event has occurred
        if (cudautils::eventIsOccurred(data.m_event->id())) {
          // It was, so data is accessible from all CUDA streams on
          // the device. Set the 'filled' for all subsequent calls and
          // return the value
          auto should_be_false = data.m_filled.exchange(true);
          assert(not should_be_false);
          data.m_fillingStream = nullptr;
        } else if (data.m_fillingStream != cudaStream) {
          // Filling is still going on. For other CUDA stream, add
          // wait on the CUDA stream and return the value. Subsequent
          // work queued on the stream will wait for the event to
          // occur (i.e. transfer to finish).
          auto ret = cudaStreamWaitEvent(cudaStream, data.m_event->id(), 0);
          cuda::throw_if_error(ret, "Failed to make a stream to wait for an event");
        }
        // else: filling is still going on. But for the same CUDA
        // stream (which would be a bit strange but fine), we can just
        // return as all subsequent work should be enqueued to the
        // same CUDA stream (or stream to be explicitly synchronized
        // by the caller)
      } else {
        // Now we can be sure that the data is not yet on the GPU, and
        // this thread is the first to try that.
        transferAsync(data.m_data, cudaStream);
        assert(data.m_fillingStream == nullptr);
        data.m_fillingStream = cudaStream;
        // Now the filling has been enqueued to the cudaStream, so we
        // can return the GPU data immediately, since all subsequent
        // work must be either enqueued to the cudaStream, or the cudaStream
        // must be synchronized by the caller
      }
    }

    return data.m_data;
  }

private:
  struct Item {
    mutable std::mutex m_mutex;
    CMS_THREAD_GUARD(m_mutex) mutable std::shared_ptr<cuda::event_t> m_event;
    CMS_THREAD_GUARD(m_mutex)
    mutable cudaStream_t m_fillingStream =
        nullptr;  // non-null if some thread is already filling (cudaStream_t is just a pointer)
    mutable std::atomic<bool> m_filled = false;  // easy check if data has been filled already or not
    CMS_THREAD_GUARD(m_mutex) mutable T m_data;
  };

  std::vector<Item> gpuDataPerDevice_;
};

#endif
