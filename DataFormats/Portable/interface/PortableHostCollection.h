#ifndef DataFormats_Portable_interface_PortableHostCollection_h
#define DataFormats_Portable_interface_PortableHostCollection_h

#include <optional>

#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

// generic SoA-based product in host memory
template <typename T>
class PortableHostCollection {
public:
  using Layout = T;
  using View = typename Layout::TrivialView;
  using Buffer = alpaka::Buf<alpaka_common::DevHost, std::byte, alpaka::DimInt<1u>, uint32_t>;

  PortableHostCollection() : buffer_{}, layout_{}, view_{} {
    // the default implementation would work correctly, but we want to add a call to the MessageLogger
    edm::LogVerbatim("PortableCollection") << __PRETTY_FUNCTION__ << " [this=" << this << "]";
  }

  PortableHostCollection(int32_t elements, alpaka_common::DevHost const &host)
      // allocate pageable host memory
      : buffer_{alpaka::allocBuf<std::byte, uint32_t>(
            host, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{Layout::computeDataSize(elements)})},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    edm::LogVerbatim("PortableCollection") << __PRETTY_FUNCTION__ << " [this=" << this << "]";
  }

  template <typename TDev>
  PortableHostCollection(int32_t elements, alpaka_common::DevHost const &host, TDev const &device)
      // allocate pinned host memory, accessible by the given device
      : buffer_{alpaka::allocMappedBuf<std::byte, uint32_t>(
            host, device, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{Layout::computeDataSize(elements)})},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    edm::LogVerbatim("PortableCollection") << __PRETTY_FUNCTION__ << " [this=" << this << "]";
  }

  ~PortableHostCollection() {
    // the default implementation would work correctly, but we want to add a call to the MessageLogger
    edm::LogVerbatim("PortableCollection") << __PRETTY_FUNCTION__ << " [this=" << this << "]";
  }

  // non-copyable
  PortableHostCollection(PortableHostCollection const &) = delete;
  PortableHostCollection &operator=(PortableHostCollection const &) = delete;

  // movable
  PortableHostCollection(PortableHostCollection &&other)
      : buffer_{std::move(other.buffer_)}, layout_{std::move(other.layout_)}, view_{std::move(other.view_)} {
    // the default implementation would work correctly, but we want to add a call to the MessageLogger
    edm::LogVerbatim("PortableCollection") << __PRETTY_FUNCTION__ << " [this=" << this << "]";
  }

  PortableHostCollection &operator=(PortableHostCollection &&other) = default;

  View &operator*() { return view_; }

  View const &operator*() const { return view_; }

  View *operator->() { return &view_; }

  View const *operator->() const { return &view_; }

  Buffer &buffer() { return *buffer_; }

  Buffer const &buffer() const { return *buffer_; }

  // part of the ROOT read streamer
  static void ROOTReadStreamer(PortableHostCollection *newObj, Layout const &layout) {
    newObj->~PortableHostCollection();
    // use the global "host" object returned by alpaka_common::host()
    new (newObj) PortableHostCollection(layout.soaMetadata().size(), alpaka_common::host());
    newObj->layout_.ROOTReadStreamer(layout);
  }

private:
  std::optional<Buffer> buffer_;  //!
  Layout layout_;                 //
  View view_;                     //!
};

#endif  // DataFormats_Portable_interface_PortableHostCollection_h
