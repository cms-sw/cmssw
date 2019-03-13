#include "catch.hpp"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

#include "test_CUDAScopedContextKernels.h"

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

namespace {
  std::unique_ptr<CUDAProduct<int *> > produce(int device, int *d, int *h) {
    auto ctx = cudatest::TestCUDAScopedContext::make(device);

    cuda::memory::async::copy(d, h, sizeof(int), ctx.stream().id());
    testCUDAScopedContextKernels_single(d, ctx.stream());
    return ctx.wrap(d);
  }
}

TEST_CASE("Use of CUDAScopedContext", "[CUDACore]") {
  exitSansCUDADevices();

  constexpr int defaultDevice = 0;
  {
    auto ctx = cudatest::TestCUDAScopedContext::make(defaultDevice);

    SECTION("Construct from device ID") {
      REQUIRE(cuda::device::current::get().id() == defaultDevice);
    }

    SECTION("Wrap T to CUDAProduct<T>") {
      std::unique_ptr<CUDAProduct<int> > dataPtr = ctx.wrap(10);
      REQUIRE(dataPtr.get() != nullptr);
      REQUIRE(dataPtr->device() == ctx.device());
      REQUIRE(dataPtr->stream().id() == ctx.stream().id());
    }

    SECTION("Construct from from CUDAProduct<T>") {
      std::unique_ptr<CUDAProduct<int>> dataPtr = ctx.wrap(10);
      const auto& data = *dataPtr;

      CUDAScopedContext ctx2{data};
      REQUIRE(cuda::device::current::get().id() == data.device());
      REQUIRE(ctx2.stream().id() == data.stream().id());
    }

    SECTION("Storing state as CUDAContextToken") {
      CUDAContextToken ctxtok;
      { // acquire
        std::unique_ptr<CUDAProduct<int>> dataPtr = ctx.wrap(10);
        const auto& data = *dataPtr;
        CUDAScopedContext ctx2{data};
        ctxtok = ctx2.toToken();
      }

      { // produce
        CUDAScopedContext ctx2{std::move(ctxtok)};
        REQUIRE(cuda::device::current::get().id() == ctx.device());
        REQUIRE(ctx2.stream().id() == ctx.stream().id());
      }
    }

    SECTION("Joining multiple CUDA streams") {
      cuda::device::current::scoped_override_t<> setDeviceForThisScope(defaultDevice);
      auto current_device = cuda::device::current::get();

      // Mimick a producer on the second CUDA stream
      int h_a1 = 1;
      auto d_a1 = cuda::memory::device::make_unique<int>(current_device);
      auto wprod1 = produce(defaultDevice, d_a1.get(), &h_a1);

      // Mimick a producer on the second CUDA stream
      int h_a2 = 2;
      auto d_a2 = cuda::memory::device::make_unique<int>(current_device);
      auto wprod2 = produce(defaultDevice, d_a2.get(), &h_a2);

      REQUIRE(wprod1->stream().id() != wprod2->stream().id());

      // Mimick a third producer "joining" the two streams
      CUDAScopedContext ctx2{*wprod1};

      auto prod1 = ctx.get(*wprod1);
      auto prod2 = ctx.get(*wprod2);

      auto d_a3 = cuda::memory::device::make_unique<int>(current_device);
      testCUDAScopedContextKernels_join(prod1, prod2, d_a3.get(), ctx.stream());
      ctx.stream().synchronize();
      REQUIRE(wprod2->event().has_occurred());

      h_a1 = 0;
      h_a2 = 0;
      int h_a3 = 0;
      cuda::memory::async::copy(&h_a1, d_a1.get(), sizeof(int), ctx.stream().id());
      cuda::memory::async::copy(&h_a2, d_a2.get(), sizeof(int), ctx.stream().id());
      cuda::memory::async::copy(&h_a3, d_a3.get(), sizeof(int), ctx.stream().id());

      REQUIRE(h_a1 == 2);
      REQUIRE(h_a2 == 4);
      REQUIRE(h_a3 == 6);
    }
  }

  // Destroy and clean up all resources so that the next test can
  // assume to start from a clean state.
  cudaCheck(cudaSetDevice(defaultDevice));
  cudaCheck(cudaDeviceSynchronize());
  cudaDeviceReset();
}
