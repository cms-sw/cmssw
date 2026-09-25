#ifndef PhysicsTools_PyTorchAlpaka_interface_Policy_h
#define PhysicsTools_PyTorchAlpaka_interface_Policy_h

#include <cstddef>
#include <optional>
#include <type_traits>
#include <cassert>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace cms::torch::alpakatools::detail {

  template <typename TDevice, typename T>
  using DeviceBuffer = cms::alpakatools::device_buffer<TDevice, T[]>;

  template <typename TQueue, typename T>
  class Policy {
  public:
    using Ttype = std::remove_const_t<T>;

    explicit Policy(T* data_ptr, const size_t num_elems) : num_elems_(num_elems), data_ptr_(data_ptr) {}

    // For constant data, create a writable copy that can be passed to
    // torch::from_blob(). For non-const data, no copy is needed.
    void copy(TQueue& queue) {
      if constexpr (std::is_const_v<T>)
        deviceToDevice(queue);
    }

    // Returns a writable pointer to a copy of constant data,
    // or the original pointer for non-const data.
    // Workaround for torch::from_blob() until pytorch supports safe COW tensors.
    Ttype* data() {
      if constexpr (std::is_const_v<T>) {
        // return buffer that can be safely used by pytorch (possibly modified)
        assert(dev_buffer_ && "DeviceBuffer not initialized! Materialize constant data first with D2D copy.");
        return dev_buffer_->data();
      } else {
        // return original pointer since it is not const
        return data_ptr_;
      }
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

    using TDevice = decltype(alpaka::getDev(std::declval<TQueue>()));

    const size_t num_elems_;
    T* data_ptr_;
    std::optional<DeviceBuffer<TDevice, Ttype>> dev_buffer_;
  };

}  // namespace cms::torch::alpakatools::detail

#endif  // PhysicsTools_PyTorchAlpaka_interface_Policy_h
