#ifndef PhysicsTools_PyTorchAlpaka_interface_Policy_h
#define PhysicsTools_PyTorchAlpaka_interface_Policy_h

#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Exception.h"

namespace cms::torch::alpakatools::detail {

  template <typename TDevice, typename T>
  using DeviceBuffer = cms::alpakatools::device_buffer<TDevice, T[]>;

  // Copy geometry in elements
  struct CopyLayout {
    size_t columns;
    size_t rows_per_column;
    size_t source_stride;
    size_t destination_stride;

    size_t destinationElements() const {
      if (columns == 0 || rows_per_column == 0)
        return 0;
      return (columns - 1) * destination_stride + rows_per_column;
    }
  };

  template <typename TQueue, typename T>
  class Policy {
  public:
    using Ttype = std::remove_const_t<T>;

    explicit Policy(T* data_ptr, CopyLayout copy_layout) : data_ptr_(data_ptr), copy_layout_(copy_layout) {}

    CopyLayout getCopyLayout() const { return copy_layout_; }

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
        if (!dev_buffer_)
          detail::throwException("Policy",
                                 "DeviceBuffer not initialized! Materialize constant data first with D2D copy.");
        return dev_buffer_->data();
      } else {
        // return original pointer since it is not const
        return data_ptr_;
      }
    }

  private:
    void deviceToDevice(TQueue& queue) {
      // lazy allocation
      const auto destination_elements = copy_layout_.destinationElements();
      if (!dev_buffer_)
        dev_buffer_ = cms::alpakatools::make_device_buffer<Ttype[]>(queue, destination_elements);
      if (destination_elements == 0)
        return;

      // Define extent and pitches for the strided 2D memcpy
      using Vec2D = alpaka::Vec<alpaka::DimInt<2>, size_t>;
      const Vec2D extent = {copy_layout_.columns, copy_layout_.rows_per_column};
      const Vec2D source_pitches{copy_layout_.source_stride * sizeof(Ttype), sizeof(Ttype)};
      const Vec2D destination_pitches{copy_layout_.destination_stride * sizeof(Ttype), sizeof(Ttype)};

      // create the views
      const auto device = alpaka::getDev(queue);
      auto source_view = alpaka::createView(device, data_ptr_, extent, source_pitches);
      auto destination_view = alpaka::createView(device, dev_buffer_->data(), extent, destination_pitches);

      alpaka::memcpy(queue, destination_view, source_view);
    }

    using TDevice = decltype(alpaka::getDev(std::declval<TQueue>()));

    T* data_ptr_;
    const CopyLayout copy_layout_;
    std::optional<DeviceBuffer<TDevice, Ttype>> dev_buffer_;
  };

}  // namespace cms::torch::alpakatools::detail

#endif  // PhysicsTools_PyTorchAlpaka_interface_Policy_h
