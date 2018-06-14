#ifndef HeterogeneousCore_CUDACore_CUDAESProduct_h
#define HeterogeneousCore_CUDACore_CUDAESProduct_h

#include <atomic>
#include <vector>

#include <cuda/api_wrappers.h>

#include "FWCore/Concurrency/interface/hardware_pause.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAServices/interface/numberOfCUDADevices.h"

template <typename T>
class CUDAESProduct {
public:
  CUDAESProduct(): gpuDataPerDevice_(numberOfCUDADevices()) {}
  ~CUDAESProduct() = default;

  // transferAsync should be a function of (T&, cuda::stream_t<>&)
  // which enqueues asynchronous transfers (possibly kernels as well)
  // to the CUDA stream
  template <typename F>
  const T& dataForCurrentDeviceAsync(cuda::stream_t<>& cudaStream, F transferAsync) const {
    edm::Service<CUDAService> cs;
    auto device = cs->getCurrentDevice();

    auto& data = gpuDataPerDevice_[device];
    if(data.m_filled.load()) {
      // GPU data has already been filled, so can return it immediately
      return data.m_data;
    }


    bool expected = false;
    if(data.m_filling.compare_exchange_strong(expected, true)) {
      // so nobody else was filling
      // then check if it got filled in the mean time
      if(data.m_filled.load()) {
        // someone else finished the filling in the meantime
        data.m_filling.store(false);
        return data.m_data;
      }
      
      // now we can be sure that the data is not yet on the GPU, and
      // this thread is the first one to try that
      try {
        transferAsync(data.m_data, cudaStream);

        cudaStream.enqueue.callback([&filling = data.m_filling,
                                     &filled = data.m_filled]
                                    (cuda::stream::id_t streamId, cuda::status_t status) mutable {
                                      // TODO: check status and throw if fails
                                      auto should_be_false = filled.exchange(true);
                                      assert(!should_be_false);
                                      auto should_be_true = filling.exchange(false);
                                      assert(should_be_true);
                                    });
      } catch(...) {
        // release the filling state and propagate exception
        auto should_be_true = data.m_filling.exchange(false);
        assert(should_be_true);
        throw std::current_exception();
      }

      // Now the filling has been enqueued to the cudaStream, so we
      // can return the GPU data immediately, since all subsequent
      // work must be either enqueued to the cudaStream, or the cudaStream
      // must be synchronized by the caller
      return data.m_data;
    }

    // can we do better than just spin on the atomic while waiting another thread to finish the filling?
    while(data.m_filling.load()) {
      hardware_pause();
    }
    assert(data.m_filled.load());

    return data.m_data;
  }
  
private:
  struct Item {
    mutable std::atomic<bool> m_filling = false; // true if some thread is already filling
    mutable std::atomic<bool> m_filled = false; // easy check if data has been filled already or not
    CMS_THREAD_GUARD(m_filling) mutable T m_data;
  };
  
  std::vector<Item> gpuDataPerDevice_;
};

#endif
