#ifndef DataFormats_PortableTest_interface_TestProductWithPtr_h
#define DataFormats_PortableTest_interface_TestProductWithPtr_h

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include <alpaka/alpaka.hpp>

/**
 * This data product is part of a test for CopyToHost::postCopy()
 * (i.e. updating a data product after the device-to-host copy). For
 * any practical purposes the indirection to 'buffer' array via the
 * 'ptr' pointer scalar is completely unnecessary. Do not take this
 * case as an example for good design of a data product.
 */
namespace portabletest {
  GENERATE_SOA_LAYOUT(TestSoALayoutWithPtr, SOA_COLUMN(int, buffer), SOA_SCALAR(int*, ptr));
  using TestSoAWithPtr = TestSoALayoutWithPtr<>;

  template <typename TDev>
  using TestProductWithPtr = PortableCollection<TDev, TestSoAWithPtr>;

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void setPtrInTestProductWithPtr(TestSoAWithPtr::View view) {
    view.ptr() = &view.buffer(0);
  }
}  // namespace portabletest

namespace cms::alpakatools {
  template <typename TDev>
  struct CopyToHost<PortableDeviceCollection<TDev, portabletest::TestSoAWithPtr>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceCollection<TDev, portabletest::TestSoAWithPtr> const& src) {
      PortableHostCollection<portabletest::TestSoAWithPtr> dst(queue, src->metadata().size());
      alpaka::memcpy(queue, dst.buffer(), src.buffer());
      return dst;
    }

    static void postCopy(PortableHostCollection<portabletest::TestSoAWithPtr>& dst) {
      portabletest::setPtrInTestProductWithPtr(dst.view());
    }
  };
}  // namespace cms::alpakatools

#endif
