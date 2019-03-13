#include "catch.hpp"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

#include <cuda_runtime_api.h>

namespace cudatest {
  class TestCUDAScopedContext {
  public:
    static
    CUDAScopedContext make(int dev) {
      auto device = cuda::device::get(dev);
      return CUDAScopedContext(dev, std::make_unique<cuda::stream_t<>>(device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream)));
    }
  };
}

TEST_CASE("Use of CUDAProduct template", "[CUDACore]") {
  SECTION("Default constructed") {
    auto foo = CUDAProduct<int>();
    REQUIRE(!foo.isValid());

    auto bar = std::move(foo);
  }

  exitSansCUDADevices();

  constexpr int defaultDevice = 0;
  {
    auto ctx = cudatest::TestCUDAScopedContext::make(defaultDevice);
    std::unique_ptr<CUDAProduct<int>> dataPtr = ctx.wrap(10);
    auto& data = *dataPtr;

    SECTION("Construct from CUDAScopedContext") {
      REQUIRE(data.isValid());
      REQUIRE(data.device() == defaultDevice);
      REQUIRE(data.stream().id() == ctx.stream().id());
      REQUIRE(&data.event() != nullptr);
    }

    SECTION("Move constructor") {
      auto data2 = CUDAProduct<int>(std::move(data));
      REQUIRE(data2.isValid());
      REQUIRE(!data.isValid());
    }

    SECTION("Move assignment") {
      CUDAProduct<int> data2;
      data2 = std::move(data);
      REQUIRE(data2.isValid());
      REQUIRE(!data.isValid());
    }
  }

  // Destroy and clean up all resources so that the next test can
  // assume to start from a clean state.
  cudaCheck(cudaSetDevice(defaultDevice));
  cudaCheck(cudaDeviceSynchronize());
  cudaDeviceReset();
}
