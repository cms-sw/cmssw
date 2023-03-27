#ifndef HeterogeneousCore_AlpakaTest_interface_AlpakaESTestData_h
#define HeterogeneousCore_AlpakaTest_interface_AlpakaESTestData_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestSoA.h"

namespace cms::alpakatest {
  // Model 1
  using AlpakaESTestDataAHost = PortableHostCollection<AlpakaESTestSoAA>;
  using AlpakaESTestDataCHost = PortableHostCollection<AlpakaESTestSoAC>;
  using AlpakaESTestDataDHost = PortableHostCollection<AlpakaESTestSoAD>;

  // Model 2
  template <typename TDev>
  class AlpakaESTestDataB {
  public:
    using Buffer = cms::alpakatools::device_buffer<TDev, int[]>;
    using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, int[]>;

    explicit AlpakaESTestDataB(Buffer buffer) : buffer_(std::move(buffer)) {}

    Buffer buffer() { return buffer_; }
    ConstBuffer buffer() const { return buffer_; }
    ConstBuffer const_buffer() const { return buffer_; }

    int const* data() const { return buffer_.data(); }
    auto size() const { return alpaka::getExtentProduct(buffer_); }

  private:
    Buffer buffer_;
  };
}  // namespace cms::alpakatest

namespace cms::alpakatools {
  template <>
  struct CopyToDevice<cms::alpakatest::AlpakaESTestDataB<alpaka_common::DevHost>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, cms::alpakatest::AlpakaESTestDataB<alpaka_common::DevHost> const& srcData) {
      // TODO: In principle associating the allocation to a queue is
      // incorrect. Framework will keep the memory alive until the IOV
      // ends. By that point all asynchronous activity using that
      // memory has finished, and the memory could be marked as "free"
      // in the allocator already by the host-side release of the
      // memory. There could also be other, independent asynchronous
      // activity going on that uses the same queue (since we don't
      // retain the queue here), and at the time of host-side release
      // the device-side release gets associated to the complemention
      // of that activity (which has nothing to do with the memory here).
      auto dstBuffer = cms::alpakatools::make_device_buffer<int[]>(queue, srcData.size());
      alpaka::memcpy(queue, dstBuffer, srcData.buffer());
      return cms::alpakatest::AlpakaESTestDataB<alpaka::Dev<TQueue>>(std::move(dstBuffer));
    }
  };
}  // namespace cms::alpakatools

#endif
