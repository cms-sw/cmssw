#ifndef HeterogeneousCore_AlpakaTest_interface_alpaka_AlpakaESTestData_h
#define HeterogeneousCore_AlpakaTest_interface_alpaka_AlpakaESTestData_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // Model 1
  class AlpakaESTestDataA {
  public:
    using Buffer = cms::alpakatools::device_buffer<Device, int[]>;
    using ConstBuffer = cms::alpakatools::const_device_buffer<Device, int[]>;

    explicit AlpakaESTestDataA(Buffer buffer) : buffer_(std::move(buffer)) {}

    Buffer buffer() { return buffer_; }
    ConstBuffer buffer() const { return buffer_; }
    ConstBuffer const_buffer() const { return buffer_; }

    int const* data() const { return buffer_.data(); }
    auto size() const { return alpaka::getExtentProduct(buffer_); }

  private:
    Buffer buffer_;
  };

  // Model 3
  using AlpakaESTestDataCHost = cms::alpakatest::AlpakaESTestDataCHost;
  using AlpakaESTestDataCDevice = PortableCollection<cms::alpakatest::AlpakaESTestSoAC>;

  using AlpakaESTestDataDHost = cms::alpakatest::AlpakaESTestDataDHost;
  using AlpakaESTestDataDDevice = PortableCollection<cms::alpakatest::AlpakaESTestSoAD>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
