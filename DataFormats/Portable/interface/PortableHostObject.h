#ifndef DataFormats_Portable_interface_PortableHostObject_h
#define DataFormats_Portable_interface_PortableHostObject_h

#include <cassert>
#include <optional>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// generic object in host memory
template <typename T>
class PortableHostObject {
public:
  using Product = T;
  using Buffer = cms::alpakatools::host_buffer<Product>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<Product>;

  PortableHostObject() = delete;

  PortableHostObject(edm::Uninitialized) noexcept {}

  PortableHostObject(alpaka_common::DevHost const& host)
      // allocate pageable host memory
      : buffer_{cms::alpakatools::make_host_buffer<Product>()}, product_{buffer_->data()} {
    assert(reinterpret_cast<uintptr_t>(product_) % alignof(Product) == 0);
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  PortableHostObject(TQueue const& queue)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<Product>(queue)}, product_{buffer_->data()} {
    assert(reinterpret_cast<uintptr_t>(product_) % alignof(Product) == 0);
  }

  // non-copyable
  PortableHostObject(PortableHostObject const&) = delete;
  PortableHostObject& operator=(PortableHostObject const&) = delete;

  // movable
  PortableHostObject(PortableHostObject&&) = default;
  PortableHostObject& operator=(PortableHostObject&&) = default;

  // default destructor
  ~PortableHostObject() = default;

  // access the product
  Product& value() { return *product_; }
  Product const& value() const { return *product_; }
  Product const& const_value() const { return *product_; }

  Product* data() { return product_; }
  Product const* data() const { return product_; }
  Product const* const_data() const { return product_; }

  Product& operator*() { return *product_; }
  Product const& operator*() const { return *product_; }

  Product* operator->() { return product_; }
  Product const* operator->() const { return product_; }

  // access the buffer
  Buffer buffer() { return *buffer_; }
  ConstBuffer buffer() const { return *buffer_; }
  ConstBuffer const_buffer() const { return *buffer_; }

  // erases the data in the Buffer by writing zeros (bytes containing '\0') to it
  void zeroInitialise() {
    std::memset(std::data(*buffer_), 0x00, alpaka::getExtentProduct(*buffer_) * sizeof(std::byte));
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  void zeroInitialise(TQueue&& queue) {
    alpaka::memset(std::forward<TQueue>(queue), *buffer_, 0x00);
  }

  // part of the ROOT read streamer
  static void ROOTReadStreamer(PortableHostObject* newObj, Product& product) {
    // destroy the default-constructed object
    newObj->~PortableHostObject();
    // use the global "host" object returned by cms::alpakatools::host()
    new (newObj) PortableHostObject(cms::alpakatools::host());
    // copy the data from the on-file object to the new one
    std::memcpy(newObj->product_, &product, sizeof(Product));
  }

private:
  std::optional<Buffer> buffer_;  //!
  Product* product_;
};

#endif  // DataFormats_Portable_interface_PortableHostObject_h
