#ifndef PhysicsTools_PyTorchAlpaka_interface_Policy_h
#define PhysicsTools_PyTorchAlpaka_interface_Policy_h

#include <cstddef>
#include <optional>
#include <type_traits>
#include <cassert>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace cms::torch::alpakatools {

  template <typename T>
  using HostBuffer = cms::alpakatools::host_buffer<T[]>;
  template <typename TDevice, typename T>
  using DeviceBuffer = cms::alpakatools::device_buffer<TDevice, T[]>;

  enum class MemcpyKind : uint8_t {
    // Copy data from host to device (used primarily for ROCm/HIP backends)
    HostToDevice = 0,
    // Copy data from device to host (used for CPU fallback inference)
    DeviceToHost = 1,
    // Copy data between device memory regions (e.g. for constant data)
    DeviceToDevice = 2,
    // Copy data between host memory regions (for CPU semantics, maps to DeviceToDevice in alpaka world).
    HostToHost = 2
  };

  template <typename TDev, typename T>
  class Policy;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

  // Provides D2D (Device-to-Device) copy support for constant data only.
  // Non-const tensors directly reuse the provided device pointer.
  template <typename T>
  class Policy<alpaka_cuda_async::Device, T> {
  public:
    using Ttype = std::remove_const_t<T>;

    explicit Policy(T* data_ptr, const size_t num_elems) : num_elems_(num_elems), data_ptr_(data_ptr) {}

    // Perform a copy operation for constant tensors.
    // Only `MemcpyKind::DeviceToDevice` is supported.
    // Non-const tensors reuse the provided pointer.
    void copy(void* queue_ptr, const MemcpyKind kind) {
      // copy only if T is const (const correctness and thread-safety -> torch::from_blob())
      if constexpr (std::is_const_v<T>) {
        auto& queue = *static_cast<alpaka_cuda_async::Queue*>(queue_ptr);
        if (kind == MemcpyKind::DeviceToDevice)
          deviceToDevice(queue);
        else
          assert(false && "Unsupported MemcpyKind, only DeviceToDevice is supported for CudaAsync backend.");
      }
    }

    // Returns a writable pointer to a copy of constant data,
    // or the original pointer for non-const data.
    // Workaround for torch::from_blob() until pytorch supports safe COW tensors.
    Ttype* data() {
      if constexpr (std::is_const_v<T>) {
        // return buffer that can be safely used by pytorch (possibly modified)
        assert(buffer_ && "DeviceBuffer not initialized! Materialize constant data first with D2D copy.");
        return alpaka::getPtrNative(buffer_.value());
      } else {
        // return original pointer since it is not const
        return data_ptr_;
      }
    }

  private:
    void deviceToDevice(alpaka_cuda_async::Queue& queue) {
      // lazy allocation
      if (!buffer_)
        buffer_ = cms::alpakatools::make_device_buffer<Ttype[]>(queue, num_elems_);
      // copy data
      auto source_view = alpaka::createView(alpaka::getDev(queue), data_ptr_, alpaka::getExtents(buffer_.value())[0]);
      alpaka::memcpy(queue, buffer_.value(), source_view);
    }

    const size_t num_elems_;
    T* data_ptr_;
    std::optional<DeviceBuffer<alpaka_cuda_async::Device, Ttype>> buffer_;
  };

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

  template <typename T>
  class Policy<alpaka_rocm_async::Device, T> {
  public:
    using Ttype = std::remove_const_t<T>;

    explicit Policy(T* data_ptr, const size_t num_elems) : num_elems_(num_elems), data_ptr_(data_ptr) {}

    // Perform a copy operation not matter if the data is constant
    // for ROCmAsync only D2H and H2D copies are considered and CPU fallback avaliable only.
    void copy(void* queue_ptr, const MemcpyKind kind) {
      auto& queue = *static_cast<alpaka_rocm_async::Queue*>(queue_ptr);
      if (kind == MemcpyKind::DeviceToHost)
        deviceToHost(queue);
      else if (kind == MemcpyKind::HostToDevice)
        hostToDevice(queue);
      else
        assert(false &&
               "Unsupported MemcpyKind, only D2H and H2D is supported for ROCmAsync backend due to CPU fallback and "
               "CMSSW pytorch-hip limitations.");
    }

    // Returns a writable pointer to a copy of constant data,
    // or the original pointer for non-const data.
    // Workaround for torch::from_blob() until pytorch supports safe COW tensors.
    Ttype* data() {
      // return buffer that can be safely used by pytorch (possibly modified)
      assert(buffer_ &&
             "HostBuffer not initialized! For ROCm/HIP CPU fallback avaliable only. Call deviceToHost first.");
      return alpaka::getPtrNative(buffer_.value());
    }

  private:
    void deviceToHost(alpaka_rocm_async::Queue& queue) {
      // lazy allocate
      if (!buffer_)
        buffer_ = cms::alpakatools::make_host_buffer<Ttype[]>(queue, num_elems_);
      // always copy data (no matter if const T* or T* for fallback)
      auto source_view = alpaka::createView(alpaka::getDev(queue), data_ptr_, alpaka::getExtents(buffer_.value())[0]);
      alpaka::memcpy(queue, buffer_.value(), source_view);
    }

    void hostToDevice(alpaka_rocm_async::Queue& queue) {
      // guard to not write to const memory space only is dest is mutable
      if constexpr (!std::is_const_v<T>) {
        assert(buffer_ &&
               "HostBuffer not initialized! For ROCm/HIP CPU fallback avaliable only. Call deviceToHost first.");
        auto dest_view = alpaka::createView(alpaka::getDev(queue), data_ptr_, alpaka::getExtents(buffer_.value())[0]);
        alpaka::memcpy(queue, dest_view, buffer_.value());
      }
    }

    const size_t num_elems_;
    T* data_ptr_;
    std::optional<HostBuffer<Ttype>> buffer_;
  };

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

  template <typename T>
  class Policy<alpaka_serial_sync::Device, T> {
  public:
    using Ttype = std::remove_const_t<T>;

    explicit Policy(T* data_ptr, const size_t num_elems) : num_elems_(num_elems), data_ptr_(data_ptr) {}

    // for SerialSync only D2D (H2H in reality but keep alpaka semantics) copy is considered
    // and only for constant data is T is not const then reuse the provided data pointer
    void copy(void* queue_ptr, const MemcpyKind kind) {
      // copy only if T is const (const correctness and thread-safety -> torch::from_blob())
      if constexpr (std::is_const_v<T>) {
        auto& queue = *static_cast<alpaka_serial_sync::Queue*>(queue_ptr);
        if (kind == MemcpyKind::DeviceToDevice || kind == MemcpyKind::HostToHost)
          hostToHost(queue);
        else
          assert(false &&
                 "Unsupported MemcpyKind, only H2H (D2D in alpaka semantics) is supported for SerialSync backend.");
      }
    }

    // Returns a writable pointer to a copy of constant data,
    // or the original pointer for non-const data.
    // Workaround for torch::from_blob() until pytorch supports safe COW tensors.
    Ttype* data() {
      if constexpr (std::is_const_v<T>) {
        // return buffer that can be safely used by pytorch (possibly modified)
        assert(buffer_ && "DeviceBuffer not initialized! Materialize constant data first with D2D copy.");
        return alpaka::getPtrNative(buffer_.value());
      } else {
        // return original pointer since it is not const
        return data_ptr_;
      }
    }

  private:
    void hostToHost(alpaka_serial_sync::Queue& queue) {
      // lazy allocation
      if (!buffer_)
        buffer_ = cms::alpakatools::make_device_buffer<Ttype[]>(queue, num_elems_);
      // copy data
      auto source_view = alpaka::createView(alpaka::getDev(queue), data_ptr_, alpaka::getExtents(buffer_.value())[0]);
      alpaka::memcpy(queue, buffer_.value(), source_view);
    }

    const size_t num_elems_;
    T* data_ptr_;
    std::optional<DeviceBuffer<alpaka_serial_sync::Device, Ttype>> buffer_;
  };

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

  template <typename T>
  class Policy<alpaka_tbb_async::Device, T> {
  public:
    using Ttype = std::remove_const_t<T>;

    explicit Policy(T* data_ptr, const size_t num_elems) : num_elems_(num_elems), data_ptr_(data_ptr) {}

    // for TbbAsync only D2D (H2H in reality but keep alpaka semantics) copy is considered
    // and only for constant data is T is not const then reuse the provided data pointer
    void copy(void* queue_ptr, const MemcpyKind kind) {
      // copy only if T is const (const correctness and thread-safety -> torch::from_blob())
      if constexpr (std::is_const_v<T>) {
        auto& queue = *static_cast<alpaka_tbb_async::Queue*>(queue_ptr);
        if (kind == MemcpyKind::DeviceToDevice || kind == MemcpyKind::HostToHost)
          hostToHost(queue);
        else
          assert(false &&
                 "Unsupported MemcpyKind, only H2H (D2D in alpaka semantics) is supported for TbbAsync backend.");
      }
    }

    // Returns a writable pointer to a copy of constant data,
    // or the original pointer for non-const data.
    // Workaround for torch::from_blob() until pytorch supports safe COW tensors.
    Ttype* data() {
      if constexpr (std::is_const_v<T>) {
        // return buffer that can be safely used by pytorch (possibly modified)
        assert(buffer_ && "DeviceBuffer not initialized! Materialize constant data first with D2D copy.");
        return alpaka::getPtrNative(buffer_.value());
      } else {
        // return original pointer since it is not const
        return data_ptr_;
      }
    }

  private:
    void hostToHost(alpaka_tbb_async::Queue& queue) {
      // lazy allocation
      if (!buffer_)
        buffer_ = cms::alpakatools::make_device_buffer<Ttype[]>(queue, num_elems_);
      // copy data
      auto source_view = alpaka::createView(alpaka::getDev(queue), data_ptr_, alpaka::getExtents(buffer_.value())[0]);
      alpaka::memcpy(queue, buffer_.value(), source_view);
    }

    const size_t num_elems_;
    T* data_ptr_;
    std::optional<DeviceBuffer<alpaka_tbb_async::Device, Ttype>> buffer_;
  };

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_Policy_h
