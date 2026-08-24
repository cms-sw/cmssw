#include "catch2/catch_all.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

TEST_CASE("copyAsync", "[cudaMemTools]") {
  if (not cms::cudatest::testDevices()) {
    return;
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  SECTION("Host to device") {
    SECTION("Single element") {
      auto host_orig = cms::cuda::make_host_unique<int>(stream);
      *host_orig = 42;

      auto device = cms::cuda::make_device_unique<int>(stream);
      auto host = cms::cuda::make_host_unique<int>(stream);

      cms::cuda::copyAsync(device, host_orig, stream);
      CUDA_CHECK(cudaMemcpyAsync(host.get(), device.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      REQUIRE(*host == 42);
    }

    SECTION("Multiple elements") {
      constexpr int N = 100;

      auto host_orig = cms::cuda::make_host_unique<int[]>(N, stream);
      for (int i = 0; i < N; ++i) {
        host_orig[i] = i;
      }

      auto device = cms::cuda::make_device_unique<int[]>(N, stream);
      auto host = cms::cuda::make_host_unique<int[]>(N, stream);

      SECTION("Copy all") {
        cms::cuda::copyAsync(device, host_orig, N, stream);
        CUDA_CHECK(cudaMemcpyAsync(host.get(), device.get(), N * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < N; ++i) {
          CHECK(host[i] == i);
        }
      }

      for (int i = 0; i < N; ++i) {
        host_orig[i] = 200 + i;
      }

      SECTION("Copy some") {
        cms::cuda::copyAsync(device, host_orig, 42, stream);
        CUDA_CHECK(cudaMemcpyAsync(host.get(), device.get(), 42 * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < 42; ++i) {
          CHECK(host[i] == 200 + i);
        }
      }
    }
  }

  SECTION("Device to host") {
    SECTION("Single element") {
      auto host_orig = cms::cuda::make_host_unique<int>(stream);
      *host_orig = 42;

      auto device = cms::cuda::make_device_unique<int>(stream);
      auto host = cms::cuda::make_host_unique<int>(stream);

      CUDA_CHECK(cudaMemcpyAsync(device.get(), host_orig.get(), sizeof(int), cudaMemcpyHostToDevice, stream));
      cms::cuda::copyAsync(host, device, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      REQUIRE(*host == 42);
    }

    SECTION("Multiple elements") {
      constexpr int N = 100;

      auto host_orig = cms::cuda::make_host_unique<int[]>(N, stream);
      for (int i = 0; i < N; ++i) {
        host_orig[i] = i;
      }

      auto device = cms::cuda::make_device_unique<int[]>(N, stream);
      auto host = cms::cuda::make_host_unique<int[]>(N, stream);

      SECTION("Copy all") {
        CUDA_CHECK(cudaMemcpyAsync(device.get(), host_orig.get(), N * sizeof(int), cudaMemcpyHostToDevice, stream));
        cms::cuda::copyAsync(host, device, N, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < N; ++i) {
          CHECK(host[i] == i);
        }
      }

      for (int i = 0; i < N; ++i) {
        host_orig[i] = 200 + i;
      }

      SECTION("Copy some") {
        CUDA_CHECK(cudaMemcpyAsync(device.get(), host_orig.get(), 42 * sizeof(int), cudaMemcpyHostToDevice, stream));
        cms::cuda::copyAsync(host, device, 42, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < 42; ++i) {
          CHECK(host[i] == 200 + i);
        }
      }
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}
