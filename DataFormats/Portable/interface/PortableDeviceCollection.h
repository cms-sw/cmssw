#ifndef DataFormats_Portable_interface_PortableDeviceCollection_h
#define DataFormats_Portable_interface_PortableDeviceCollection_h

#include <cassert>
#include <optional>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// generic SoA-based product in device memory
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
class PortableDeviceCollection {
  static_assert(not std::is_same_v<TDev, alpaka_common::DevHost>,
                "Use PortableHostCollection<T> instead of PortableDeviceCollection<T, DevHost>");

public:
  using Layout = T;
  using View = typename Layout::View;
  using ConstView = typename Layout::ConstView;
  using Buffer = cms::alpakatools::device_buffer<TDev, std::byte[]>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, std::byte[]>;

  PortableDeviceCollection() = default;

  PortableDeviceCollection(int32_t elements, TDev const& device)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, Layout::computeDataSize(elements))},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  PortableDeviceCollection(int32_t elements, TQueue const& queue)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, Layout::computeDataSize(elements))},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // non-copyable
  PortableDeviceCollection(PortableDeviceCollection const&) = delete;
  PortableDeviceCollection& operator=(PortableDeviceCollection const&) = delete;

  // movable
  PortableDeviceCollection(PortableDeviceCollection&&) = default;
  PortableDeviceCollection& operator=(PortableDeviceCollection&&) = default;

  // default destructor
  ~PortableDeviceCollection() = default;

  // access the View
  View& view() { return view_; }
  ConstView const& view() const { return view_; }
  ConstView const& const_view() const { return view_; }

  View& operator*() { return view_; }
  ConstView const& operator*() const { return view_; }

  View* operator->() { return &view_; }
  ConstView const* operator->() const { return &view_; }

  // access the Buffer
  Buffer buffer() { return *buffer_; }
  ConstBuffer buffer() const { return *buffer_; }
  ConstBuffer const_buffer() const { return *buffer_; }

private:
  std::optional<Buffer> buffer_;  //!
  Layout layout_;                 //
  View view_;                     //!
};

#endif  // DataFormats_Portable_interface_PortableDeviceCollection_h
