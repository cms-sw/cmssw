#include <optional>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/moveToDeviceAsync.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// each test binary is built for a single Alpaka backend
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {
  template <typename T>
  class TestHostBuffer {
  public:
    using Buffer = cms::alpakatools::host_buffer<T[]>;
    using ConstBuffer = cms::alpakatools::const_host_buffer<T[]>;

    template <typename TQueue>
    TestHostBuffer(TQueue const& queue, int size) : buffer_(cms::alpakatools::make_host_buffer<T[]>(queue, size)) {}

    TestHostBuffer(TestHostBuffer const&) = delete;
    TestHostBuffer& operator=(TestHostBuffer const&) = delete;
    ;
    TestHostBuffer(TestHostBuffer&& other) {
      buffer_ = std::move(*other.buffer_);
      other.buffer_.reset();
    }
    TestHostBuffer& operator=(TestHostBuffer& other) {
      buffer_ = std::move(*other.buffer_);
      other.buffer_.reset();
      return this;
    }

    bool has_value() const { return buffer_.has_value(); }

    T* data() { return buffer_->data(); }

    Buffer buffer() { return *buffer_; }
    ConstBuffer buffer() const { return *buffer_; }

  private:
    std::optional<Buffer> buffer_;
  };

  template <typename T, typename TDev>
  class TestDeviceBuffer {
  public:
    using Buffer = cms::alpakatools::device_buffer<TDev, T[]>;

    template <typename TQueue>
    TestDeviceBuffer(TQueue const& queue, int size) : buffer_(cms::alpakatools::make_device_buffer<T[]>(queue, size)) {}

    T* data() { return buffer_.data(); }

    Buffer buffer() { return buffer_; }

  private:
    Buffer buffer_;
  };

  template <typename T>
  void fillBuffer(TestHostBuffer<T>& buffer) {
    for (int i = 0, size = alpaka::getExtentProduct(buffer.buffer()); i < size; ++i) {
      buffer.data()[i] = i;
    }
  }
}  // namespace

namespace cms::alpakatools {
  template <typename T>
  struct CopyToDevice<TestHostBuffer<T>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, TestHostBuffer<T> const& hostBuffer) {
      TestDeviceBuffer<T, alpaka::Dev<TQueue>> deviceBuffer(queue, alpaka::getExtentProduct(hostBuffer.buffer()));
      alpaka::memcpy(queue, deviceBuffer.buffer(), hostBuffer.buffer());
      return deviceBuffer;
    }
  };
}  // namespace cms::alpakatools

TEST_CASE("Test moveToDeviceAsync() for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  // run the test on each device
  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    auto queue = Queue(device);
    constexpr int size = 32;
    TestHostBuffer<int> buffer_host(queue, size);
    fillBuffer(buffer_host);
    auto const* ptr_host = buffer_host.data();

    auto buffer_device = cms::alpakatools::moveToDeviceAsync(queue, std::move(buffer_host));
    REQUIRE(not buffer_host.has_value());
    if constexpr (std::is_same_v<Device, alpaka_common::DevHost>) {
      REQUIRE(buffer_device.data() == ptr_host);
    } else {
      REQUIRE(buffer_device.data() != ptr_host);
    }
    alpaka::exec<Acc1D>(
        queue,
        cms::alpakatools::make_workdiv<Acc1D>(1, size),
        [] ALPAKA_FN_ACC(Acc1D const& acc, int const* data) {
          for (int i : cms::alpakatools::uniform_elements(acc)) {
            assert(data[i] == i);
          }
        },
        buffer_device.data());
    alpaka::wait(queue);

    /* the following should not compile
    auto buffer2_host = cms::alpakatools::make_host_buffer<int>();
    auto buffer2_device = cms::alpakatools::moveToDeviceAsync(queue, std::move(buffer2_host));
    */
  }
}
