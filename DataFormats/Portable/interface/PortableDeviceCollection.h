#ifndef DataFormats_Portable_interface_PortableDeviceCollection_h
#define DataFormats_Portable_interface_PortableDeviceCollection_h

#include <cassert>
#include <concepts>
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

// generic SoA-based product in device memory
template <typename TDev, typename T0, typename... Args>
class PortableDeviceMultiCollection {
  //static_assert(alpaka::isDevice<TDev>);
  static_assert(not std::is_same_v<TDev, alpaka_common::DevHost>,
                "Use PortableHostCollection<T> instead of PortableDeviceCollection<T, DevHost>");

  template <typename T>
  static constexpr std::size_t count_t_ = portablecollection::typeCount<T, T0, Args...>;

  template <typename T>
  static constexpr std::size_t index_t_ = portablecollection::typeIndex<T, T0, Args...>;

  static constexpr std::size_t members_ = sizeof...(Args) + 1;

public:
  using Buffer = cms::alpakatools::device_buffer<TDev, std::byte[]>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, std::byte[]>;
  using Implementation = portablecollection::CollectionImpl<0, T0, Args...>;

  using SizesArray = std::array<int32_t, members_>;

  template <std::size_t Idx = 0>
  using Layout = portablecollection::TypeResolver<Idx, T0, Args...>;

  //template <std::size_t Idx = 0>
  //using View = typename Layout<Idx>::View;
  // Workaround for flaky expansion of tempaltes by nvcc (expanding with "Args" instead of "Args...
  template <std::size_t Idx = 0UL>
  using View = typename std::tuple_element<Idx, std::tuple<T0, Args...>>::type::View;

  //template <std::size_t Idx = 0>
  //using ConstView = typename Layout<Idx>::ConstView;
  // Workaround for flaky expansion of tempaltes by nvcc (expanding with "Args" instead of "Args..."
  template <std::size_t Idx = 0UL>
  using ConstView = typename std::tuple_element<Idx, std::tuple<T0, Args...>>::type::ConstView;

private:
  template <std::size_t Idx>
  using Leaf = portablecollection::CollectionLeaf<Idx, Layout<Idx>>;

  template <std::size_t Idx>
  Leaf<Idx>& get() {
    return static_cast<Leaf<Idx>&>(impl_);
  }

  template <std::size_t Idx>
  Leaf<Idx> const& get() const {
    return static_cast<Leaf<Idx> const&>(impl_);
  }

  template <typename T>
  Leaf<index_t_<T>>& get() {
    return static_cast<Leaf<index_t_<T>>&>(impl_);
  }

  template <typename T>
  Leaf<index_t_<T>> const& get() const {
    return static_cast<Leaf<index_t_<T>> const&>(impl_);
  }

public:
  PortableDeviceMultiCollection() = delete;

  explicit PortableDeviceMultiCollection(edm::Uninitialized) noexcept {};

  PortableDeviceMultiCollection(int32_t elements, TDev const& device)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  PortableDeviceMultiCollection(int32_t elements, TQueue const& queue)
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  static int32_t computeDataSize(const SizesArray& sizes) {
    int32_t ret = 0;
    portablecollection::constexpr_for<0, members_>(
        [&sizes, &ret](auto i) { ret += Layout<i>::computeDataSize(sizes[i]); });
    return ret;
  }

  PortableDeviceMultiCollection(const SizesArray& sizes, TDev const& device)
      // allocate device memory
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, computeDataSize(sizes))},
        impl_{buffer_->data(), sizes} {
    portablecollection::constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    portablecollection::constexpr_for<1, members_>(
        [&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  PortableDeviceMultiCollection(const SizesArray& sizes, TQueue const& queue)
      // allocate device memory asynchronously on the given work queue
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, computeDataSize(sizes))},
        impl_{buffer_->data(), sizes} {
    portablecollection::constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    portablecollection::constexpr_for<1, members_>(
        [&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  // non-copyable
  PortableDeviceMultiCollection(PortableDeviceMultiCollection const&) = delete;
  PortableDeviceMultiCollection& operator=(PortableDeviceMultiCollection const&) = delete;

  // movable
  PortableDeviceMultiCollection(PortableDeviceMultiCollection&&) = default;
  PortableDeviceMultiCollection& operator=(PortableDeviceMultiCollection&&) = default;

  // default destructor
  ~PortableDeviceMultiCollection() = default;

  // access the View by index
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>& view() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& const_view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>& operator*() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& operator*() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>* operator->() {
    return &get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const* operator->() const {
    return &get<Idx>().view_;
  }

  // access the View by type
  template <typename T>
  typename T::View& view() {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& view() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& const_view() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::View& operator*() {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& operator*() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::View* operator->() {
    return &get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const* operator->() const {
    return &get<T>().view_;
  }

  // access the Buffer
  Buffer buffer() { return *buffer_; }
  ConstBuffer buffer() const { return *buffer_; }
  ConstBuffer const_buffer() const { return *buffer_; }

  // erases the data in the Buffer by writing zeros (bytes containing '\0') to it
  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  void zeroInitialise(TQueue&& queue) {
    alpaka::memset(std::forward<TQueue>(queue), *buffer_, 0x00);
  }

  // extract the sizes array
  SizesArray sizes() const {
    SizesArray ret;
    portablecollection::constexpr_for<0, members_>([&](auto i) { ret[i] = get<i>().layout_.metadata().size(); });
    return ret;
  }

private:
  std::optional<Buffer> buffer_;  //!
  Implementation impl_;           // (serialized: this is where the layouts live)
};

#endif  // DataFormats_Portable_interface_PortableDeviceCollection_h
