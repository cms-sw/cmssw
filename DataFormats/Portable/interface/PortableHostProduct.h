#ifndef DataFormats_Portable_interface_PortableHostProduct_h
#define DataFormats_Portable_interface_PortableHostProduct_h

#include <cassert>
#include <optional>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// generic SoA-based product in host memory
template <typename T>
class PortableHostProduct {
public:
  using Product = T;
  using Buffer = cms::alpakatools::host_buffer<Product>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<Product>;

  PortableHostProduct() = default;

  PortableHostProduct(alpaka_common::DevHost const& host)
      // allocate pageable host memory
      : buffer_{cms::alpakatools::make_host_buffer<Product>()}, product_{buffer_->data()} {
    assert(reinterpret_cast<uintptr_t>(product_) % alignof(Product) == 0);
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  PortableHostProduct(TQueue const& queue)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<Product>(queue)}, product_{buffer_->data()} {
    assert(reinterpret_cast<uintptr_t>(product_) % alignof(Product) == 0);
  }

  // non-copyable
  PortableHostProduct(PortableHostProduct const&) = delete;
  PortableHostProduct& operator=(PortableHostProduct const&) = delete;

  // movable
  PortableHostProduct(PortableHostProduct&&) = default;
  PortableHostProduct& operator=(PortableHostProduct&&) = default;

  // default destructor
  ~PortableHostProduct() = default;

  // access the product
  Product& value() { return *product_; }
  Product const& value() const { return *product_; }

  Product* data() { return product_; }
  Product const* data() const { return product_; }

  Product& operator*() { return *product_; }
  Product const& operator*() const { return *product_; }

  Product* operator->() { return product_; }
  Product const* operator->() const { return product_; }

  // access the buffer
  Buffer buffer() { return *buffer_; }
  ConstBuffer buffer() const { return *buffer_; }
  ConstBuffer const_buffer() const { return *buffer_; }

  // part of the ROOT read streamer
  static void ROOTReadStreamer(PortableHostProduct* newObj, Product& product) {
    std::cerr << "ROOT object at    " << &product << std::endl;
    std::cerr << "id:  " << product.id << std::endl;

    std::cerr << "CMSSW object at   " << newObj << std::endl;
    std::cerr << "buffer present ?  " << newObj->buffer_.has_value() << std::endl;
    if (newObj->buffer_.has_value()) {
      std::cerr << "buffer content at " << newObj->buffer_->data() << std::endl;
    }
    std::cerr << "struct content at " << newObj->product_ << std::endl;
    if (newObj->product_) {
      std::cerr << "id:  " << newObj->product_->id << std::endl;
    }
    newObj->~PortableHostProduct();
    // use the global "host" object returned by cms::alpakatools::host()
    new (newObj) PortableHostProduct(cms::alpakatools::host());
    std::memcpy(newObj->product_, &product, sizeof(Product));
    std::cerr << "CMSSW object at   " << newObj << std::endl;
    std::cerr << "buffer present ?  " << newObj->buffer_.has_value() << std::endl;
    if (newObj->buffer_.has_value()) {
      std::cerr << "buffer content at " << newObj->buffer_->data() << std::endl;
    }
    std::cerr << "struct content at " << newObj->product_ << std::endl;
    if (newObj->product_) {
      std::cerr << "id:  " << newObj->product_->id << std::endl;
    }
  }

private:
  std::optional<Buffer> buffer_;  //!
  Product* product_;
};

#endif  // DataFormats_Portable_interface_PortableHostProduct_h
