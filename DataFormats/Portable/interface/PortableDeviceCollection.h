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
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// generic SoA-based product in device memory
template <typename TDev, typename T, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
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

  template <std::integral Int>
  PortableDeviceCollection(TDev const& device, const Int size)
    requires(!requires { Layout::blocksNumber; })
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(
            device, Layout::computeDataSize(portablecollection::size_cast(size)))},
        layout_{buffer_->data(), portablecollection::size_cast(size)},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  template <typename TQueue, std::integral Int>
    requires(alpaka::isQueue<TQueue> && (!requires { Layout::blocksNumber; }))
  PortableDeviceCollection(TQueue const& queue, const Int size)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(
            queue, Layout::computeDataSize(portablecollection::size_cast(size)))},
        layout_{buffer_->data(), portablecollection::size_cast(size)},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
  template <std::integral... Ints>
  explicit PortableDeviceCollection(TDev const& device, const Ints... sizes)
    requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
      : PortableDeviceCollection(device, std::to_array({portablecollection::size_cast(sizes)...})) {}

  // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
  template <typename TQueue, std::integral... Ints>
    requires(alpaka::isQueue<TQueue>)
  explicit PortableDeviceCollection(TQueue const& queue, const Ints... sizes)
    requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
      : PortableDeviceCollection(queue, std::to_array({portablecollection::size_cast(sizes)...})) {}

  // constructor for a SoABlocks-layout, taking per-block sizes as a fixed-size array
  template <std::size_t N>
  explicit PortableDeviceCollection(TDev const& device, std::array<int32_t, N> const& sizes)
    requires requires { Layout::blocksNumber; } && (N == static_cast<std::size_t>(Layout::blocksNumber))
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, Layout::computeDataSize(sizes))},
        layout_{buffer_->data(), sizes},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // constructor for a SoABlocks-layout, taking per-block sizes as a fixed-size array
  template <typename TQueue, std::size_t N>
    requires(alpaka::isQueue<TQueue>)
  explicit PortableDeviceCollection(TQueue const& queue, std::array<int32_t, N> const& sizes)
    requires requires { Layout::blocksNumber; } && (N == static_cast<std::size_t>(Layout::blocksNumber))
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, Layout::computeDataSize(sizes))},
        layout_{buffer_->data(), sizes},
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

  // erases the data in the Buffer by writing zeros (bytes containing '\0') to it
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)
  void zeroInitialise(TQueue&& queue) {
    alpaka::memset(std::forward<TQueue>(queue), *buffer_, 0x00);
  }

  // Copy column by column heterogeneously for device to host/device data transfer.
  // TODO: implement heterogeneous deepCopy for SoA blocks
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue> && (!requires { Layout::blocksNumber; }))
  void deepCopy(TQueue& queue, ConstView const& view) {
    ConstDescriptor desc{view};
    Descriptor desc_{view_};
    _deepCopy<0>(queue, desc_, desc);
  }

  // Either Layout::size_type for normal layouts or std::array<Layout::size_type, N> for SoABlocks layouts
  auto size() const { return layout_.metadata().size(); }

private:
  // Helper function implementing the recursive deep copy
  template <int I, typename TQueue>
  void _deepCopy(TQueue& queue, Descriptor& dest, ConstDescriptor const& src) {
    if constexpr (I < ConstDescriptor::num_cols) {
      assert(std::get<I>(dest.buff).size_bytes() == std::get<I>(src.buff).size_bytes());
      alpaka::memcpy(
          queue,
          alpaka::createView(alpaka::getDev(queue), std::get<I>(dest.buff).data(), std::get<I>(dest.buff).size()),
          alpaka::createView(alpaka::getDev(queue), std::get<I>(src.buff).data(), std::get<I>(src.buff).size()));
      _deepCopy<I + 1>(queue, dest, src);
    }
  }

  // Data members
  std::optional<Buffer> buffer_;  //!
  Layout layout_;                 //
  View view_;                     //!
};

namespace ngt {

  // Specialize MemoryCopyTraits for PortableDeviceCollection
  template <typename T, typename TDev>
  struct MemoryCopyTraits<PortableDeviceCollection<T, TDev>> {
    using value_type = PortableDeviceCollection<T, TDev>;

    // Properties are the collection size: T::size_type, or std::array<T::size_type, N> for SoABlocks.
    using Properties = decltype(std::declval<value_type>()->metadata().size());

    static Properties properties(value_type const& object) { return object->metadata().size(); }

    template <typename TQueue>
    static void initialize(TQueue& queue, value_type& object, Properties const& size)
      requires(alpaka::isQueue<TQueue>)
    {
      object = value_type(queue, size);
    }

    static std::vector<std::span<std::byte>> regions(value_type& object) {
      std::byte* address = reinterpret_cast<std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}};
    }

    static std::vector<std::span<const std::byte>> regions(value_type const& object) {
      const std::byte* address = reinterpret_cast<const std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}};
    }
  };
}  // namespace ngt

#endif  // DataFormats_Portable_interface_PortableDeviceCollection_h
