#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

TEST_CASE("memsetAsync", "[cudaMemTools]") {
  exitSansCUDADevices();

  auto current_device = cuda::device::current::get();
  auto stream = current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);

  SECTION("Single element") {
    auto host_orig = cudautils::make_host_unique<int>(stream);
    *host_orig = 42;

    auto device = cudautils::make_device_unique<int>(stream);
    auto host = cudautils::make_host_unique<int>(stream);
    cudautils::copyAsync(device, host_orig, stream);
    cudautils::memsetAsync(device, 0, stream);
    cudautils::copyAsync(host, device, stream);
    stream.synchronize();

    REQUIRE(*host == 0);
  }

  SECTION("Multiple elements") {
    constexpr int N = 100;

    auto host_orig = cudautils::make_host_unique<int[]>(N, stream);
    for (int i = 0; i < N; ++i) {
      host_orig[i] = i;
    }

    auto device = cudautils::make_device_unique<int[]>(N, stream);
    auto host = cudautils::make_host_unique<int[]>(N, stream);
    cudautils::copyAsync(device, host_orig, N, stream);
    cudautils::memsetAsync(device, 0, N, stream);
    cudautils::copyAsync(host, device, N, stream);
    stream.synchronize();

    for (int i = 0; i < N; ++i) {
      CHECK(host[i] == 0);
    }
  }
}
