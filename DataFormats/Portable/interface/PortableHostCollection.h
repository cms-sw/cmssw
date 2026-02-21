#ifndef DataFormats_Portable_interface_PortableHostCollection_h
#define DataFormats_Portable_interface_PortableHostCollection_h

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <optional>
#include <type_traits>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableCollectionCommon.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// generic SoA-based product in host memory
template <typename T>
class PortableHostCollection {
public:
  using Layout = T;
  using View = typename Layout::View;
  using ConstView = typename Layout::ConstView;
  using Descriptor = typename Layout::Descriptor;
  using ConstDescriptor = typename Layout::ConstDescriptor;
  using Buffer = cms::alpakatools::host_buffer<std::byte[]>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<std::byte[]>;

  PortableHostCollection() = delete;

  explicit PortableHostCollection(edm::Uninitialized) noexcept {}

  template <std::integral Int>
  PortableHostCollection(alpaka_common::DevHost const& host, const Int size)
    requires(!requires { Layout::blocksNumber; })
      // allocate pageable host memory
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(
            Layout::computeDataSize(portablecollection::size_cast(size)))},
        layout_{buffer_->data(), portablecollection::size_cast(size)},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  template <typename TQueue, std::integral Int>
    requires(alpaka::isQueue<TQueue> && (!requires { Layout::blocksNumber; }))
  PortableHostCollection(TQueue const& queue, const Int size)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(
            queue, Layout::computeDataSize(portablecollection::size_cast(size)))},
        layout_{buffer_->data(), portablecollection::size_cast(size)},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // constructor for code that does not use alpaka explicitly, using the global "host" object returned by cms::alpakatools::host()
  template <std::integral Int>
  PortableHostCollection(const Int size)
    requires(!requires { Layout::blocksNumber; })
      : PortableHostCollection(cms::alpakatools::host(), size) {}

  // constructor for code that does not use alpaka explicitly, using the global "host" object returned by cms::alpakatools::host()
  // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
  template <std::integral... Ints>
  PortableHostCollection(const Ints... sizes)
    requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
      : PortableHostCollection(cms::alpakatools::host(), std::to_array({portablecollection::size_cast(sizes)...})) {}

  // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
  template <std::integral... Ints>
  explicit PortableHostCollection(alpaka_common::DevHost const& host, const Ints... sizes)
    requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
      // allocate pageable host memory
      : PortableHostCollection(host, std::to_array({portablecollection::size_cast(sizes)...})) {}

  // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
  template <typename TQueue, std::integral... Ints>
    requires(alpaka::isQueue<TQueue>)
  explicit PortableHostCollection(TQueue const& queue, const Ints... sizes)
    requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : PortableHostCollection(queue, std::to_array({portablecollection::size_cast(sizes)...})) {}

  // constructor for a SoABlocks-layout, taking per-block sizes as a fixed-size array
  template <std::size_t N>
  explicit PortableHostCollection(alpaka_common::DevHost const& host, std::array<int32_t, N> const& sizes)
    requires requires { Layout::blocksNumber; } && (N == static_cast<std::size_t>(Layout::blocksNumber))
      // allocate pageable host memory
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(Layout::computeDataSize(sizes))},
        layout_{buffer_->data(), sizes},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // constructor for a SoABlocks-layout, taking per-block sizes as a fixed-size array
  template <typename TQueue, std::size_t N>
    requires(alpaka::isQueue<TQueue>)
  explicit PortableHostCollection(TQueue const& queue, std::array<int32_t, N> const& sizes)
    requires requires { Layout::blocksNumber; } && (N == static_cast<std::size_t>(Layout::blocksNumber))
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, Layout::computeDataSize(sizes))},
        layout_{buffer_->data(), sizes},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // non-copyable
  PortableHostCollection(PortableHostCollection const&) = delete;
  PortableHostCollection& operator=(PortableHostCollection const&) = delete;

  // movable
  PortableHostCollection(PortableHostCollection&&) = default;
  PortableHostCollection& operator=(PortableHostCollection&&) = default;

  // default destructor
  ~PortableHostCollection() = default;

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
  void zeroInitialise() {
    std::memset(std::data(*buffer_), 0x00, alpaka::getExtentProduct(*buffer_) * sizeof(std::byte));
  }

  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)
  void zeroInitialise(TQueue&& queue) {
    alpaka::memset(std::forward<TQueue>(queue), *buffer_, 0x00);
  }

  // part of the ROOT read streamer
  static void ROOTReadStreamer(PortableHostCollection* newObj, Layout& layout) {
    // destroy the default-constructed collection
    newObj->~PortableHostCollection();

    // construct in-place a new collection, with the known size, using the global "host" object returned by cms::alpakatools::host()
    new (newObj) PortableHostCollection(cms::alpakatools::host(), layout.metadata().size());

    // copy the data from the on-file layout to the new collection
    newObj->layout_.ROOTReadStreamer(layout);
    // free the memory allocated by ROOT
    layout.ROOTStreamerCleaner();
  }

  // Copy column by column the content of the given ConstView into this PortableHostCollection.
  // The view must point to data in host memory.
  void deepCopy(ConstView const& view) { layout_.deepCopy(view); }

  // Copy column by column heterogeneously for device to host data transfer.
  // TODO: implement heterogeneous deepCopy for SoA blocks
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue> && (!requires { Layout::blocksNumber; }))
  void deepCopy(TQueue& queue, ConstView const& view) {
    ConstDescriptor desc{view};
    Descriptor desc_{view_};
    portablecollection::deepCopy<0>(queue, desc_, desc);
  }

  // Either Layout::size_type for normal layouts or std::array<Layout::size_type, N> for SoABlocks layouts
  auto size() const { return layout_.metadata().size(); }

private:
  // Data members
  std::optional<Buffer> buffer_;  //!
  Layout layout_;                 //
  View view_;                     //!
};

namespace ngt {

  // Specialize the MemoryCopyTraits for PortableHostColletion
  template <typename T>
  struct MemoryCopyTraits<PortableHostCollection<T>> {
    using value_type = PortableHostCollection<T>;
    // For basic Layouts -> T::size_type, for Layouts by blocks -> std::array<T::size_type, N>
    using Properties = decltype(std::declval<value_type>()->metadata().size());

    // The properties needed to initialize a new PrortableHostCollection are just its size.
    static Properties properties(value_type const& object) { return object->metadata().size(); }

    // Replace the default-constructed empty object with one where the buffer has been allocated in pageable system memory.
    static void initialize(value_type& object, Properties const& size) {
      object = value_type(cms::alpakatools::host(), size);
    }

    static std::vector<std::span<std::byte>> regions(value_type& object) {
      // The whole PortableHostCollection is stored in a single contiguous memory region.
      std::byte* address = reinterpret_cast<std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}};
    }

    static std::vector<std::span<const std::byte>> regions(value_type const& object) {
      // The whole PortableHostCollection is stored in a single contiguous memory region.
      const std::byte* address = reinterpret_cast<const std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}};
    }
  };
}  // namespace ngt

#endif  // DataFormats_Portable_interface_PortableHostCollection_h
