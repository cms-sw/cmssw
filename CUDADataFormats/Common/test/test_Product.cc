#include "catch.hpp"

#include "CUDADataFormats/Common/interface/Product.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/StreamCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/EventCache.h"

#include <cuda_runtime_api.h>

namespace cms::cudatest {
  class TestScopedContext {
  public:
    static cuda::ScopedContextProduce make(int dev, bool createEvent) {
      cms::cuda::SharedEventPtr event;
      if (createEvent) {
        event = cms::cuda::getEventCache().get();
      }
      return cuda::ScopedContextProduce(dev, cms::cuda::getStreamCache().get(), std::move(event));
    }
  };
}  // namespace cms::cudatest

TEST_CASE("Use of cms::cuda::Product template", "[CUDACore]") {
  SECTION("Default constructed") {
    auto foo = cms::cuda::Product<int>();
    REQUIRE(!foo.isValid());

    auto bar = std::move(foo);
  }

  if (not cms::cudatest::testDevices()) {
    return;
  }

  constexpr int defaultDevice = 0;
  cudaCheck(cudaSetDevice(defaultDevice));
  {
    auto ctx = cms::cudatest::TestScopedContext::make(defaultDevice, true);
    std::unique_ptr<cms::cuda::Product<int>> dataPtr = ctx.wrap(10);
    auto& data = *dataPtr;

    SECTION("Construct from cms::cuda::ScopedContext") {
      REQUIRE(data.isValid());
      REQUIRE(data.device() == defaultDevice);
      REQUIRE(data.stream() == ctx.stream());
      REQUIRE(data.event() != nullptr);
    }

    SECTION("Move constructor") {
      auto data2 = cms::cuda::Product<int>(std::move(data));
      REQUIRE(data2.isValid());
      REQUIRE(!data.isValid());
    }

    SECTION("Move assignment") {
      cms::cuda::Product<int> data2;
      data2 = std::move(data);
      REQUIRE(data2.isValid());
      REQUIRE(!data.isValid());
    }
  }

  cudaCheck(cudaSetDevice(defaultDevice));
  cudaCheck(cudaDeviceSynchronize());
  // Note: CUDA resources are cleaned up by the destructors of the global cache objects
}
