#ifndef CUDADataFormats_Common_interface_PortableHostCollection_h
#define CUDADataFormats_Common_interface_PortableHostCollection_h

#include <cassert>
#include <cstdlib>

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

namespace cms::cuda {

  // generic SoA-based product in host memory
  template <typename T>
  class PortableHostCollection {
  public:
    using Layout = T;
    using View = typename Layout::View;
    using ConstView = typename Layout::ConstView;
    using Buffer = cms::cuda::host::unique_ptr<std::byte[]>;

    PortableHostCollection() = default;

    PortableHostCollection(int32_t elements)
        // allocate pageable host memory
        : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout::computeDataSize(elements))},
          layout_{buffer_.get(), elements},
          view_{layout_} {
      // make_host_unique for pageable host memory uses a default alignment of 128 bytes
      assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
    }

    PortableHostCollection(int32_t elements, cudaStream_t stream)
        // allocate pinned host memory, accessible by the current device
        : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout::computeDataSize(elements), stream)},
          layout_{buffer_.get(), elements},
          view_{layout_} {
      // CUDA pinned host memory uses a default alignment of at least 128 bytes
      assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
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
    Buffer& buffer() { return buffer_; }
    Buffer const& buffer() const { return buffer_; }
    Buffer const& const_buffer() const { return buffer_; }

    size_t bufferSize() const { return layout_.metadata().byteSize(); }

    // part of the ROOT read streamer
    static void ROOTReadStreamer(PortableHostCollection* newObj, Layout const& layout) {
      newObj->~PortableHostCollection();
      // allocate pinned host memory using the legacy stream, that synchronises with all (blocking) streams
      new (newObj) PortableHostCollection(layout.metadata().size());
      newObj->layout_.ROOTReadStreamer(layout);
    }

  private:
    Buffer buffer_;  //!
    Layout layout_;  //
    View view_;      //!
  };

}  // namespace cms::cuda

#endif  // CUDADataFormats_Common_interface_PortableHostCollection_h
