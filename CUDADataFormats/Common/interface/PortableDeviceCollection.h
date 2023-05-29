#ifndef CUDADataFormats_Common_interface_PortableDeviceCollection_h
#define CUDADataFormats_Common_interface_PortableDeviceCollection_h

#include <cassert>
#include <cstdlib>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

namespace cms::cuda {

  // generic SoA-based product in device memory
  template <typename T>
  class PortableDeviceCollection {
  public:
    using Layout = T;
    using View = typename Layout::View;
    using ConstView = typename Layout::ConstView;
    using Buffer = cms::cuda::device::unique_ptr<std::byte[]>;

    PortableDeviceCollection() = default;

    PortableDeviceCollection(int32_t elements, cudaStream_t stream)
        : buffer_{cms::cuda::make_device_unique<std::byte[]>(Layout::computeDataSize(elements), stream)},
          layout_{buffer_.get(), elements},
          view_{layout_} {
      // CUDA device memory uses a default alignment of at least 128 bytes
      assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
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
    Buffer& buffer() { return buffer_; }
    Buffer const& buffer() const { return buffer_; }
    Buffer const& const_buffer() const { return buffer_; }

    size_t bufferSize() const { return layout_.metadata().byteSize(); }

  private:
    Buffer buffer_;  //!
    Layout layout_;  //
    View view_;      //!
  };

}  // namespace cms::cuda

#endif  // CUDADataFormats_Common_interface_PortableDeviceCollection_h
