#ifndef DataFormats_Portable_interface_PortableDeviceCollection_h
#define DataFormats_Portable_interface_PortableDeviceCollection_h

#include <optional>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// generic SoA-based product in device memory
template <typename T, typename TDev>
class PortableDeviceCollection {
  static_assert(not std::is_same_v<TDev, alpaka_common::DevHost>,
                "Use PortableHostCollection<T> instead of PortableDeviceCollection<T, DevHost>");

public:
  using Layout = T;
  using Buffer = alpaka::Buf<TDev, std::byte, alpaka::DimInt<1u>, uint32_t>;

  PortableDeviceCollection() = default;

  PortableDeviceCollection(int32_t elements, TDev const &device)
      : buffer_{alpaka::allocBuf<std::byte, uint32_t>(
            device, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{Layout::compute_size(elements)})},
        layout_{elements, buffer_->data()} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    alpaka::pin(*buffer_);
  }

  ~PortableDeviceCollection() = default;

  // non-copyable
  PortableDeviceCollection(PortableDeviceCollection const &) = delete;
  PortableDeviceCollection &operator=(PortableDeviceCollection const &) = delete;

  // movable
  PortableDeviceCollection(PortableDeviceCollection &&other) = default;
  PortableDeviceCollection &operator=(PortableDeviceCollection &&other) = default;

  Layout &operator*() { return layout_; }

  Layout const &operator*() const { return layout_; }

  Layout *operator->() { return &layout_; }

  Layout const *operator->() const { return &layout_; }

  Buffer &buffer() { return *buffer_; }

  Buffer const &buffer() const { return *buffer_; }

private:
  std::optional<Buffer> buffer_;  //!
  Layout layout_;                 //
};

#endif  // DataFormats_Portable_interface_PortableDeviceCollection_h
