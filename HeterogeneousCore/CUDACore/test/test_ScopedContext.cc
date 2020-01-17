#include "catch.hpp"

#include "CUDADataFormats/Common/interface/Product.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/eventWorkHasCompleted.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/StreamCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/EventCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"
#include "HeterogeneousCore/CUDAUtilities/interface/ScopedSetDevice.h"

#include "test_ScopedContextKernels.h"

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

namespace {
  std::unique_ptr<cms::cuda::Product<int*>> produce(int device, int* d, int* h) {
    auto ctx = cms::cudatest::TestScopedContext::make(device, true);
    cudaCheck(cudaMemcpyAsync(d, h, sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
    cms::cudatest::testScopedContextKernels_single(d, ctx.stream());
    return ctx.wrap(d);
  }
}  // namespace

TEST_CASE("Use of cms::cuda::ScopedContext", "[CUDACore]") {
  if (not cms::cudatest::testDevices()) {
    return;
  }

  constexpr int defaultDevice = 0;
  {
    auto ctx = cms::cudatest::TestScopedContext::make(defaultDevice, true);

    SECTION("Construct from device ID") { REQUIRE(cms::cuda::currentDevice() == defaultDevice); }

    SECTION("Wrap T to cms::cuda::Product<T>") {
      std::unique_ptr<cms::cuda::Product<int>> dataPtr = ctx.wrap(10);
      REQUIRE(dataPtr.get() != nullptr);
      REQUIRE(dataPtr->device() == ctx.device());
      REQUIRE(dataPtr->stream() == ctx.stream());
    }

    SECTION("Construct from from cms::cuda::Product<T>") {
      std::unique_ptr<cms::cuda::Product<int>> dataPtr = ctx.wrap(10);
      const auto& data = *dataPtr;

      cms::cuda::ScopedContextProduce ctx2{data};
      REQUIRE(cms::cuda::currentDevice() == data.device());
      REQUIRE(ctx2.stream() == data.stream());

      // Second use of a product should lead to new stream
      cms::cuda::ScopedContextProduce ctx3{data};
      REQUIRE(cms::cuda::currentDevice() == data.device());
      REQUIRE(ctx3.stream() != data.stream());
    }

    SECTION("Storing state in cms::cuda::ContextState") {
      cms::cuda::ContextState ctxstate;
      {  // acquire
        std::unique_ptr<cms::cuda::Product<int>> dataPtr = ctx.wrap(10);
        const auto& data = *dataPtr;
        edm::WaitingTaskWithArenaHolder dummy{
            edm::make_waiting_task(tbb::task::allocate_root(), [](std::exception_ptr const* iPtr) {})};
        cms::cuda::ScopedContextAcquire ctx2{data, std::move(dummy), ctxstate};
      }

      {  // produce
        cms::cuda::ScopedContextProduce ctx2{ctxstate};
        REQUIRE(cms::cuda::currentDevice() == ctx.device());
        REQUIRE(ctx2.stream() == ctx.stream());
      }
    }

    SECTION("Joining multiple CUDA streams") {
      cms::cuda::ScopedSetDevice setDeviceForThisScope(defaultDevice);

      // Mimick a producer on the first CUDA stream
      int h_a1 = 1;
      auto d_a1 = cms::cuda::make_device_unique<int>(nullptr);
      auto wprod1 = produce(defaultDevice, d_a1.get(), &h_a1);

      // Mimick a producer on the second CUDA stream
      int h_a2 = 2;
      auto d_a2 = cms::cuda::make_device_unique<int>(nullptr);
      auto wprod2 = produce(defaultDevice, d_a2.get(), &h_a2);

      REQUIRE(wprod1->stream() != wprod2->stream());

      // Mimick a third producer "joining" the two streams
      cms::cuda::ScopedContextProduce ctx2{*wprod1};

      auto prod1 = ctx2.get(*wprod1);
      auto prod2 = ctx2.get(*wprod2);

      auto d_a3 = cms::cuda::make_device_unique<int>(nullptr);
      cms::cudatest::testScopedContextKernels_join(prod1, prod2, d_a3.get(), ctx2.stream());
      cudaCheck(cudaStreamSynchronize(ctx2.stream()));
      REQUIRE(wprod2->isAvailable());
      REQUIRE(cms::cuda::eventWorkHasCompleted(wprod2->event()));

      h_a1 = 0;
      h_a2 = 0;
      int h_a3 = 0;

      cudaCheck(cudaMemcpyAsync(&h_a1, d_a1.get(), sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
      cudaCheck(cudaMemcpyAsync(&h_a2, d_a2.get(), sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
      cudaCheck(cudaMemcpyAsync(&h_a3, d_a3.get(), sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));

      REQUIRE(h_a1 == 2);
      REQUIRE(h_a2 == 4);
      REQUIRE(h_a3 == 6);
    }
  }

  cudaCheck(cudaSetDevice(defaultDevice));
  cudaCheck(cudaDeviceSynchronize());
  // Note: CUDA resources are cleaned up by the destructors of the global cache objects
}
