#ifndef PhysicsTools_PyTorchAlpaka_interface_Policy_h
#define PhysicsTools_PyTorchAlpaka_interface_Policy_h

#include <cstddef>
#include <optional>
#include <type_traits>
#include <cassert>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace cms::torch::alpakatools::detail {

  template <typename T>
  using HostBuffer = cms::alpakatools::host_buffer<T[]>;
  template <typename TDevice, typename T>
  using DeviceBuffer = cms::alpakatools::device_buffer<TDevice, T[]>;

  enum class MemcpyKind : uint8_t {
    // Copy data from host to device (used for ROCm/HIP backends)
    HostToDevice = 0,
    // Copy data from device to host (used for CPU fallback inference)
    DeviceToHost = 1,
    // Copy data between device memory regions (e.g. for constant data)
    DeviceToDevice = 2,
    // Copy data between host memory regions (for CPU semantics, maps to DeviceToDevice in alpaka world).
    HostToHost = 2
  };

  template <typename TQueue, typename T>
  class Policy {
  public:
    using Ttype = std::remove_const_t<T>;

    explicit Policy(T* data_ptr, const size_t num_elems) : num_elems_(num_elems), data_ptr_(data_ptr) {}

    // Perform a copy operation for constant tensors.
    // For CudaAsync, SerialSync only D2D (H2H for CPU but stick to alpaka semantics) copy is considered
    // and only for constant data. If T is not const then reuse the provided data pointer.
    //
    // For ROCmAsync only D2H and H2D copies are considered and CPU fallback avaliable only.
    // Perform a copy operation not matter if the data is constant
    void copy(TQueue& queue, const MemcpyKind kind) {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      if (kind == MemcpyKind::DeviceToHost)
        deviceToHost(queue);
      else if (kind == MemcpyKind::HostToDevice)
        hostToDevice(queue);
      else
        assert(
            false &&
            "Unsupported MemcpyKind, only D2H and H2D copy is supported for ROCmAsync backend due to CPU fallback and "
            "CMSSW pytorch-hip limitations.");
#else
      // copy only if T is const (const correctness and thread-safety -> torch::from_blob())
      if constexpr (std::is_const_v<T>) {
        if (kind == MemcpyKind::DeviceToDevice)
          deviceToDevice(queue);
        else
          assert(false && "Unsupported MemcpyKind, only D2D copy operation is supported this backend.");
      }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
    }

    // Returns a writable pointer to a copy of constant data,
    // or the original pointer for non-const data.
    // Workaround for torch::from_blob() until pytorch supports safe COW tensors.
    Ttype* data() {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      // return buffer that can be safely used by pytorch (possibly modified)
      assert(host_buffer_ &&
             "HostBuffer not initialized! For ROCm/HIP CPU fallback avaliable only. Call deviceToHost first.");
      return host_buffer_->data();
#else
      if constexpr (std::is_const_v<T>) {
        // return buffer that can be safely used by pytorch (possibly modified)
        assert(dev_buffer_ && "DeviceBuffer not initialized! Materialize constant data first with D2D copy.");
        return dev_buffer_->data();
      } else {
        // return original pointer since it is not const
        return data_ptr_;
      }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
    }

  private:
    void deviceToDevice(TQueue& queue) {
      // lazy allocation
      if (!dev_buffer_)
        dev_buffer_ = cms::alpakatools::make_device_buffer<Ttype[]>(queue, num_elems_);
      // copy data
      auto source_view =
          alpaka::createView(alpaka::getDev(queue), data_ptr_, alpaka::getExtents(dev_buffer_.value())[0]);
      alpaka::memcpy(queue, dev_buffer_.value(), source_view);
    }

    void deviceToHost(TQueue& queue) {
      // lazy allocate
      if (!host_buffer_)
        host_buffer_ = cms::alpakatools::make_host_buffer<Ttype[]>(queue, num_elems_);
      // always copy data (no matter if const T* or T* for fallback)
      auto source_view =
          alpaka::createView(alpaka::getDev(queue), data_ptr_, alpaka::getExtents(host_buffer_.value())[0]);
      alpaka::memcpy(queue, host_buffer_.value(), source_view);
    }

    void hostToDevice(TQueue& queue) {
      // guard to not write to const memory space only is dest is mutable
      if constexpr (!std::is_const_v<T>) {
        assert(host_buffer_ &&
               "HostBuffer not initialized! For ROCm/HIP CPU fallback avaliable only. Call deviceToHost first.");
        auto dest_view =
            alpaka::createView(alpaka::getDev(queue), data_ptr_, alpaka::getExtents(host_buffer_.value())[0]);
        alpaka::memcpy(queue, dest_view, host_buffer_.value());
      }
    }

    using TDevice = decltype(alpaka::getDev(std::declval<TQueue>()));

    const size_t num_elems_;
    T* data_ptr_;
    std::optional<HostBuffer<Ttype>> host_buffer_;
    std::optional<DeviceBuffer<TDevice, Ttype>> dev_buffer_;
  };

}  // namespace cms::torch::alpakatools::detail

#endif  // PhysicsTools_PyTorchAlpaka_interface_Policy_h
