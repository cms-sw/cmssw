#ifndef DataFormats_Portable_interface_PortableDeviceObject_h
#define DataFormats_Portable_interface_PortableDeviceObject_h

#include <cassert>
#include <optional>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// generic object in device memory
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
class PortableDeviceObject {
  static_assert(not std::is_same_v<TDev, alpaka_common::DevHost>,
                "Use PortableHostObject<T> instead of PortableDeviceObject<T, DevHost>");

public:
  using Product = T;
  using Buffer = cms::alpakatools::device_buffer<TDev, Product>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, Product>;

  PortableDeviceObject() = delete;

  PortableDeviceObject(edm::Uninitialized) {}

  PortableDeviceObject(TDev const& device)
      // allocate global device memory
      : buffer_{cms::alpakatools::make_device_buffer<Product>(device)} {
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % alignof(Product) == 0);
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  PortableDeviceObject(TQueue const& queue)
      // allocate global device memory with queue-ordered semantic
      : buffer_{cms::alpakatools::make_device_buffer<Product>(queue)} {
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % alignof(Product) == 0);
  }

  // non-copyable
  PortableDeviceObject(PortableDeviceObject const&) = delete;
  PortableDeviceObject& operator=(PortableDeviceObject const&) = delete;

  // movable
  PortableDeviceObject(PortableDeviceObject&&) = default;
  PortableDeviceObject& operator=(PortableDeviceObject&&) = default;

  // default destructor
  ~PortableDeviceObject() = default;

  // access the product
  Product& value() { return *buffer_->data(); }
  Product const& value() const { return *buffer_->data(); }
  Product const& const_value() const { return *buffer_->data(); }

  Product* data() { return buffer_->data(); }
  Product const* data() const { return buffer_->data(); }
  Product const* const_data() const { return buffer_->data(); }

  Product& operator*() { return *buffer_->data(); }
  Product const& operator*() const { return *buffer_->data(); }

  Product* operator->() { return buffer_->data(); }
  Product const* operator->() const { return buffer_->data(); }

  // access the buffer
  Buffer buffer() { return *buffer_; }
  ConstBuffer buffer() const { return *buffer_; }
  ConstBuffer const_buffer() const { return *buffer_; }

  // erases the data in the Buffer by writing zeros (bytes containing '\0') to it
  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  void zeroInitialise(TQueue&& queue) {
    alpaka::memset(std::forward<TQueue>(queue), *buffer_, 0x00);
  }

private:
  std::optional<Buffer> buffer_;
};

#endif  // DataFormats_Portable_interface_PortableDeviceObject_h
