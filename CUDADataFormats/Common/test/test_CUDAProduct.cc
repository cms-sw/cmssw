#include "catch.hpp"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAStreamCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAEventCache.h"

#include <cuda_runtime_api.h>

namespace cudatest {
  class TestCUDAScopedContext {
  public:
    static CUDAScopedContextProduce make(int dev, bool createEvent) {
      cudautils::SharedEventPtr event;
      if (createEvent) {
        event = cudautils::getCUDAEventCache().getCUDAEvent();
      }
      return CUDAScopedContextProduce(dev, cudautils::getCUDAStreamCache().getCUDAStream(), std::move(event));
    }
  };
}  // namespace cudatest

TEST_CASE("Use of CUDAProduct template", "[CUDACore]") {
  SECTION("Default constructed") {
    auto foo = CUDAProduct<int>();
    REQUIRE(!foo.isValid());

    auto bar = std::move(foo);
  }

  exitSansCUDADevices();

  constexpr int defaultDevice = 0;
  cudaCheck(cudaSetDevice(defaultDevice));
  {
    auto ctx = cudatest::TestCUDAScopedContext::make(defaultDevice, true);
    std::unique_ptr<CUDAProduct<int>> dataPtr = ctx.wrap(10);
    auto& data = *dataPtr;

    SECTION("Construct from CUDAScopedContext") {
      REQUIRE(data.isValid());
      REQUIRE(data.device() == defaultDevice);
      REQUIRE(data.stream() == ctx.stream());
      REQUIRE(data.event() != nullptr);
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

  cudaCheck(cudaSetDevice(defaultDevice));
  cudaCheck(cudaDeviceSynchronize());
  // Note: CUDA resources are cleaned up by the destructors of the global cache objects
}
