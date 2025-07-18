#ifndef HeterogeneousCore_AlpakaTest_interface_AlpakaESTestData_h
#define HeterogeneousCore_AlpakaTest_interface_AlpakaESTestData_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestSoA.h"

namespace cms::alpakatest {
  // PortableCollection-based model
  using AlpakaESTestDataAHost = PortableHostCollection<AlpakaESTestSoAA>;
  using AlpakaESTestDataCHost = PortableHostCollection<AlpakaESTestSoAC>;
  using AlpakaESTestDataDHost = PortableHostCollection<AlpakaESTestSoAD>;

  using AlpakaESTestDataACMultiHost = PortableHostMultiCollection<AlpakaESTestSoAA, AlpakaESTestSoAC>;

  // Template-over-device model
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

  // Template-over-device model with PortableCollection members
  // Demonstrates indirection from one PortableCollection to the other
  template <typename TDev>
  class AlpakaESTestDataE {
  public:
    using ECollection = PortableCollection<AlpakaESTestSoAE, TDev>;
    using EDataCollection = PortableCollection<AlpakaESTestSoAEData, TDev>;

    class ConstView {
    public:
      constexpr ConstView(typename ECollection::ConstView e, typename EDataCollection::ConstView data)
          : eView_(e), dataView_(data) {}

      constexpr auto size() const { return eView_.metadata().size(); }
      constexpr int val(int i) const { return eView_.val(i); }
      constexpr int val2(int i) const { return dataView_.val2(eView_.ind(i)); }

    private:
      typename ECollection::ConstView eView_;
      typename EDataCollection::ConstView dataView_;
    };

    AlpakaESTestDataE(size_t size, size_t dataSize) : e_(size), data_(dataSize) {}

    AlpakaESTestDataE(ECollection e, EDataCollection data) : e_(std::move(e)), data_(std::move(data)) {}

    ECollection const& e() const { return e_; }
    EDataCollection const& data() const { return data_; }

    ConstView view() const { return const_view(); }
    ConstView const_view() const { return ConstView(e_.const_view(), data_.const_view()); }

  private:
    ECollection e_;
    EDataCollection data_;
  };
  using AlpakaESTestDataEHost = AlpakaESTestDataE<alpaka_common::DevHost>;

}  // namespace cms::alpakatest

namespace cms::alpakatools {
  // Explicit specializations are needed for the template-over-device model
  //
  // PortableCollection-based model gets these for free from PortableCollection itself

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

  template <>
  struct CopyToDevice<cms::alpakatest::AlpakaESTestDataEHost> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, cms::alpakatest::AlpakaESTestDataEHost const& srcData) {
      using ECopy = CopyToDevice<cms::alpakatest::AlpakaESTestDataEHost::ECollection>;
      using EDataCopy = CopyToDevice<cms::alpakatest::AlpakaESTestDataEHost::EDataCollection>;
      using TDevice = alpaka::Dev<TQueue>;
      return cms::alpakatest::AlpakaESTestDataE<TDevice>(ECopy::copyAsync(queue, srcData.e()),
                                                         EDataCopy::copyAsync(queue, srcData.data()));
    }
  };
}  // namespace cms::alpakatools

#endif
