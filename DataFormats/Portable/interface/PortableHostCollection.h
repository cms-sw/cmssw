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
// TODO: Should we define a size alias for the number of elements? Also should be switch to unsigned and/or 64 bit?
// Exeeding the size of int32_t fails without clear error messages.
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

  PortableHostCollection(int32_t elements, alpaka_common::DevHost const& host)
    requires(!portablecollection::hasBlocksNumber<Layout>)
      // allocate pageable host memory
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(Layout::computeDataSize(elements))},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  template <typename TQueue>
    requires(alpaka::isQueue<TQueue> && (!portablecollection::hasBlocksNumber<Layout>))
  PortableHostCollection(int32_t elements, TQueue const& queue)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, Layout::computeDataSize(elements))},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // constructor for code that does not use alpaka explicitly, using the global "host" object returned by cms::alpakatools::host()
  PortableHostCollection(int32_t elements) : PortableHostCollection(elements, cms::alpakatools::host()) {}
  // constructor for SoA by blocks with a variadic of sizes

  template <std::integral... Ints>
    requires(portablecollection::hasBlocksNumber<Layout>)
  explicit PortableHostCollection(alpaka_common::DevHost const& host, Ints... sizes)
    requires(sizeof...(sizes) == Layout::blocksNumber)
      // allocate pageable host memory
      : PortableHostCollection(host, std::to_array({static_cast<int32_t>(sizes)...})) {}

  // constructor for SoA by blocks with a variadic of sizes
  template <typename TQueue, std::integral... Ints>
    requires(alpaka::isQueue<TQueue> && portablecollection::hasBlocksNumber<Layout>)
  explicit PortableHostCollection(TQueue const& queue, Ints... sizes)
    requires(sizeof...(sizes) == Layout::blocksNumber)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : PortableHostCollection(queue, std::to_array({static_cast<int32_t>(sizes)...})) {}

  // constructor for SoA by blocks with an array of sizes
  template <std::size_t N>
    requires(portablecollection::hasBlocksNumber<Layout>)
  explicit PortableHostCollection(alpaka_common::DevHost const& host, std::array<int32_t, N> const& sizes)
      // allocate pageable host memory
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(Layout::computeDataSize(sizes))},
        layout_{buffer_->data(), sizes},
        view_{layout_} {
    static_assert(Layout::blocksNumber == N, "Number of sizes must match the number of blocks in the Layout");
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  // constructor for SoA by blocks with an array of sizes
  template <typename TQueue, std::size_t N>
    requires(alpaka::isQueue<TQueue> && portablecollection::hasBlocksNumber<Layout>)
  explicit PortableHostCollection(TQueue const& queue, std::array<int32_t, N> const& sizes)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, Layout::computeDataSize(sizes))},
        layout_{buffer_->data(), sizes},
        view_{layout_} {
    static_assert(Layout::blocksNumber == N, "Number of sizes must match the number of blocks in the Layout");
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

  static void ROOTReadStreamer(PortableHostCollection* newObj, Layout& layout) {
    // destroy the default-constructed collection
    newObj->~PortableHostCollection();

    // construct in-place a new collection, with the known size, using the global "host" object returned by cms::alpakatools::host()
    if constexpr (portablecollection::hasBlocksNumber<Layout>) {
      // Version with blocks: (host, size)
      new (newObj) PortableHostCollection(cms::alpakatools::host(), layout.metadata().size());
    } else {
      // Version without blocks: (size, host)
      new (newObj) PortableHostCollection(layout.metadata().size(), cms::alpakatools::host());
    }

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
    requires(alpaka::isQueue<TQueue> && (!portablecollection::hasBlocksNumber<Layout>))
  void deepCopy(ConstView const& view, TQueue& queue) {
    ConstDescriptor desc{view};
    Descriptor desc_{view_};
    _deepCopy<0>(desc_, desc, queue);
  }

  // Either int32_t for normal layouts or std::array<int32_t, N> for SoABlocks layouts
  auto size() const {
    return layout_.metadata().size();
    ;
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

namespace ngt {

  // Specialize the MemoryCopyTraits for PortableHostColletion
  template <typename T>
  struct MemoryCopyTraits<PortableHostCollection<T>> {
    using value_type = PortableHostCollection<T>;
    // For basic Layouts -> T::size_type, for Layouts by blocks -> std::array<T::size_type, N>
    using Properties = decltype(std::declval<value_type>()->metadata().size());

    // The properties needed to initialize a new PrortableHostCollection are just its size.
    static Properties properties(value_type const& object) { return object->metadata().size(); }

    static void initialize(value_type& object, Properties const& size)
      requires(!portablecollection::hasBlocksNumber<T>)
    {
      object = value_type(size, cms::alpakatools::host());
    }

    static void initialize(value_type& object, Properties const& size)
      requires portablecollection::hasBlocksNumber<T>
    {
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
