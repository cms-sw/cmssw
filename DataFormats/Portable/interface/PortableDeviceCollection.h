#ifndef DataFormats_Portable_interface_PortableDeviceCollection_h
#define DataFormats_Portable_interface_PortableDeviceCollection_h

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableCollectionCommon.h"
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
  using Descriptor = typename Layout::Descriptor;
  using ConstDescriptor = typename Layout::ConstDescriptor;
  using Buffer = cms::alpakatools::device_buffer<TDev, std::byte[]>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, std::byte[]>;

  PortableDeviceCollection() = delete;

  explicit PortableDeviceCollection(edm::Uninitialized) noexcept {}

  PortableDeviceCollection(int32_t elements, TDev const& device)
    requires(!portablecollection::hasBlocksNumber<Layout>)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, Layout::computeDataSize(elements))},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  template <typename TQueue>
    requires(alpaka::isQueue<TQueue> && (!portablecollection::hasBlocksNumber<Layout>))
  PortableDeviceCollection(int32_t elements, TQueue const& queue)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, Layout::computeDataSize(elements))},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // constructor for SoA by blocks with a variadic of sizes
  template <std::integral... Ints>
    requires(portablecollection::hasBlocksNumber<Layout>)
  explicit PortableDeviceCollection(TDev const& device, Ints... sizes)
    requires(sizeof...(sizes) == Layout::blocksNumber)
      : PortableDeviceCollection(device, std::to_array({static_cast<int32_t>(sizes)...})) {}

  // constructor for SoA by blocks with a variadic of sizes
  template <typename TQueue, std::integral... Ints>
    requires(alpaka::isQueue<TQueue> && portablecollection::hasBlocksNumber<Layout>)
  explicit PortableDeviceCollection(TQueue const& queue, Ints... sizes)
    requires(sizeof...(sizes) == Layout::blocksNumber)
      : PortableDeviceCollection(queue, std::to_array({static_cast<int32_t>(sizes)...})) {}

  // constructor for SoA by blocks with an array of sizes
  template <std::size_t N>
    requires(portablecollection::hasBlocksNumber<Layout>)
  explicit PortableDeviceCollection(TDev const& device, std::array<int32_t, N> const& sizes)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, Layout::computeDataSize(sizes))},
        layout_{buffer_->data(), sizes},
        view_{layout_} {
    static_assert(Layout::blocksNumber == N, "Number of sizes must match the number of blocks in the Layout");
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // constructor for SoA by blocks with an array of sizes
  template <typename TQueue, std::size_t N>
    requires(alpaka::isQueue<TQueue> && portablecollection::hasBlocksNumber<Layout>)
  explicit PortableDeviceCollection(TQueue const& queue, std::array<int32_t, N> const& sizes)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, Layout::computeDataSize(sizes))},
        layout_{buffer_->data(), sizes},
        view_{layout_} {
    static_assert(Layout::blocksNumber == N, "Number of sizes must match the number of blocks in the Layout");
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

  // erases the data in the Buffer by writing zeros (bytes containing '\0') to it
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)
  void zeroInitialise(TQueue&& queue) {
    alpaka::memset(std::forward<TQueue>(queue), *buffer_, 0x00);
  }

  // Copy column by column heterogeneously for device to host/device data transfer.
  // TODO: implement heterogeneous deepCopy for SoA blocks
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue> && (!portablecollection::hasBlocksNumber<Layout>))
  void deepCopy(ConstView const& view, TQueue& queue) {
    ConstDescriptor desc{view};
    Descriptor desc_{view_};
    _deepCopy<0>(desc_, desc, queue);
  }

private:
  // Helper function implementing the recursive deep copy
  template <int I, typename TQueue>
  void _deepCopy(Descriptor& dest, ConstDescriptor const& src, TQueue& queue) {
    if constexpr (I < ConstDescriptor::num_cols) {
      assert(std::get<I>(dest.buff).size_bytes() == std::get<I>(src.buff).size_bytes());
      alpaka::memcpy(
          queue,
          alpaka::createView(alpaka::getDev(queue), std::get<I>(dest.buff).data(), std::get<I>(dest.buff).size()),
          alpaka::createView(alpaka::getDev(queue), std::get<I>(src.buff).data(), std::get<I>(src.buff).size()));
      _deepCopy<I + 1>(dest, src, queue);
    }
  }

  // Data members
  std::optional<Buffer> buffer_;  //!
  Layout layout_;                 //
  View view_;                     //!
};

#endif  // DataFormats_Portable_interface_PortableDeviceCollection_h
