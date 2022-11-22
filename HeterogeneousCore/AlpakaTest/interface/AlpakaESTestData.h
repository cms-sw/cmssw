#ifndef HeterogeneousCore_AlpakaTest_interface_AlpakaESTestData_h
#define HeterogeneousCore_AlpakaTest_interface_AlpakaESTestData_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestSoA.h"

namespace cms::alpakatest {
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

  // Model 3
  using AlpakaESTestDataCHost = PortableHostCollection<AlpakaESTestSoAC>;
  using AlpakaESTestDataDHost = PortableHostCollection<AlpakaESTestSoAD>;
}  // namespace cms::alpakatest

#endif
