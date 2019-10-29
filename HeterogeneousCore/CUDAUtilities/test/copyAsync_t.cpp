#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

TEST_CASE("copyAsync", "[cudaMemTools]") {
  exitSansCUDADevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreate(&stream));

  SECTION("Host to device") {
    SECTION("Single element") {
      auto host_orig = cudautils::make_host_unique<int>(stream);
      *host_orig = 42;

      auto device = cudautils::make_device_unique<int>(stream);
      auto host = cudautils::make_host_unique<int>(stream);

      cudautils::copyAsync(device, host_orig, stream);
      cudaCheck(cudaMemcpyAsync(host.get(), device.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaStreamSynchronize(stream));

      REQUIRE(*host == 42);
    }

    SECTION("Multiple elements") {
      constexpr int N = 100;

      auto host_orig = cudautils::make_host_unique<int[]>(N, stream);
      for (int i = 0; i < N; ++i) {
        host_orig[i] = i;
      }

      auto device = cudautils::make_device_unique<int[]>(N, stream);
      auto host = cudautils::make_host_unique<int[]>(N, stream);

      SECTION("Copy all") {
        cudautils::copyAsync(device, host_orig, N, stream);
        cudaCheck(cudaMemcpyAsync(host.get(), device.get(), N * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        for (int i = 0; i < N; ++i) {
          CHECK(host[i] == i);
        }
      }

      for (int i = 0; i < N; ++i) {
        host_orig[i] = 200 + i;
      }

      SECTION("Copy some") {
        cudautils::copyAsync(device, host_orig, 42, stream);
        cudaCheck(cudaMemcpyAsync(host.get(), device.get(), 42 * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        for (int i = 0; i < 42; ++i) {
          CHECK(host[i] == 200 + i);
        }
      }
    }
  }

  SECTION("Device to host") {
    SECTION("Single element") {
      auto host_orig = cudautils::make_host_unique<int>(stream);
      *host_orig = 42;

      auto device = cudautils::make_device_unique<int>(stream);
      auto host = cudautils::make_host_unique<int>(stream);

      cudaCheck(cudaMemcpyAsync(device.get(), host_orig.get(), sizeof(int), cudaMemcpyHostToDevice, stream));
      cudautils::copyAsync(host, device, stream);
      cudaCheck(cudaStreamSynchronize(stream));

      REQUIRE(*host == 42);
    }

    SECTION("Multiple elements") {
      constexpr int N = 100;

      auto host_orig = cudautils::make_host_unique<int[]>(N, stream);
      for (int i = 0; i < N; ++i) {
        host_orig[i] = i;
      }

      auto device = cudautils::make_device_unique<int[]>(N, stream);
      auto host = cudautils::make_host_unique<int[]>(N, stream);

      SECTION("Copy all") {
        cudaCheck(cudaMemcpyAsync(device.get(), host_orig.get(), N * sizeof(int), cudaMemcpyHostToDevice, stream));
        cudautils::copyAsync(host, device, N, stream);
        cudaCheck(cudaStreamSynchronize(stream));
        for (int i = 0; i < N; ++i) {
          CHECK(host[i] == i);
        }
      }

      for (int i = 0; i < N; ++i) {
        host_orig[i] = 200 + i;
      }

      SECTION("Copy some") {
        cudaCheck(cudaMemcpyAsync(device.get(), host_orig.get(), 42 * sizeof(int), cudaMemcpyHostToDevice, stream));
        cudautils::copyAsync(host, device, 42, stream);
        cudaCheck(cudaStreamSynchronize(stream));
        for (int i = 0; i < 42; ++i) {
          CHECK(host[i] == 200 + i);
        }
      }
    }
  }

  cudaCheck(cudaStreamDestroy(stream));
}
