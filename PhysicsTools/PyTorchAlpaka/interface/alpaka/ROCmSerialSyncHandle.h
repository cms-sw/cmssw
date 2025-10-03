#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_ROCmSerialSyncHandle_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_ROCmSerialSyncHandle_h

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace alpaka_rocm_async::torch {

  using namespace cms::alpakatools;

  // Helper class to provide SerialSync backend fallback on ROCmAsync based modules
  template <typename T>
  class ROCmSerialSyncHandle {
  public:
    explicit ROCmSerialSyncHandle(const void* device_ptr, const size_t ncols, const size_t nelems)
        : device_ptr_(device_ptr), extent_(Vec1D{ncols * nelems}), h_buf_(make_host_buffer<T[]>(ncols * nelems)) {}

    // Synchronization responsibility move to the caller
    void copyToHost(Queue& queue) {
      auto d_view =
          alpaka::createView(alpaka::getDev(queue), const_cast<T*>(static_cast<const T*>(device_ptr_)), extent_);
      alpaka::memcpy(queue, h_buf_.value(), d_view);
    }

    // Synchronization responsibility move to the caller
    void copyToDevice(Queue& queue) {
      auto d_view =
          alpaka::createView(alpaka::getDev(queue), const_cast<T*>(static_cast<const T*>(device_ptr_)), extent_);
      alpaka::memcpy(queue, d_view, h_buf_.value());
    }

    const void* ptr() const { return alpaka::getPtrNative(h_buf_.value()); }

  private:
    const void* device_ptr_;
    Vec1D extent_;
    std::optional<host_buffer<T[]>> h_buf_;
  };

}  // namespace alpaka_rocm_async::torch

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_ROCmSerialSyncHandle_h