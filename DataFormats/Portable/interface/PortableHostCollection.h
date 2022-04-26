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
  using Buffer = alpaka::Buf<alpaka_common::DevHost, std::byte, alpaka::DimInt<1u>, uint32_t>;

  PortableHostCollection() = default;

  PortableHostCollection(int32_t elements, alpaka_common::DevHost const &host)
      // allocate pageable host memory
      : buffer_{alpaka::allocBuf<std::byte, uint32_t>(
            host, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{Layout::compute_size(elements)})},
        layout_{elements, buffer_->data()} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  template <typename TDev>
  PortableHostCollection(int32_t elements, alpaka_common::DevHost const &host, TDev const &device)
      // allocate pinned host memory, accessible by the given device
      : buffer_{alpaka::allocMappedBuf<std::byte, uint32_t>(
            host, device, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{Layout::compute_size(elements)})},
        layout_{elements, buffer_->data()} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
  }

  ~PortableHostCollection() = default;

  // non-copyable
  PortableHostCollection(PortableHostCollection const &) = delete;
  PortableHostCollection &operator=(PortableHostCollection const &) = delete;

  // movable
  PortableHostCollection(PortableHostCollection &&other) = default;
  PortableHostCollection &operator=(PortableHostCollection &&other) = default;

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

#endif  // DataFormats_Portable_interface_PortableHostCollection_h
