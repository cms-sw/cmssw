#include "catch.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

namespace {
  CUDAService makeCUDAService(edm::ParameterSet ps, edm::ActivityRegistry& ar) {
    auto desc = edm::ConfigurationDescriptions("Service", "CUDAService");
    CUDAService::fillDescriptions(desc);
    desc.validate(ps, "CUDAService");
    return CUDAService(ps, ar);
  }
}

TEST_CASE("memsetAsync", "[cudaMemTools]") {
  exitSansCUDADevices();

  edm::ActivityRegistry ar;
  edm::ParameterSet ps;
  auto cs = makeCUDAService(ps, ar);

  auto current_device = cuda::device::current::get();
  auto stream = current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);

  SECTION("Single element") {
    auto host_orig = cs.make_host_unique<int>(stream);
    *host_orig = 42;

    auto device = cs.make_device_unique<int>(stream);
    auto host = cs.make_host_unique<int>(stream);
    cudautils::copyAsync(device, host_orig, stream);
    cudautils::memsetAsync(device, 0, stream);
    cudautils::copyAsync(host, device, stream);
    stream.synchronize();

    REQUIRE(*host == 0);
  }

  SECTION("Multiple elements") {
    constexpr int N = 100;

    auto host_orig = cs.make_host_unique<int[]>(N, stream);
    for(int i=0; i<N; ++i) {
      host_orig[i] = i;
    }

    auto device = cs.make_device_unique<int[]>(N, stream);
    auto host = cs.make_host_unique<int[]>(N, stream);
    cudautils::copyAsync(device, host_orig, N, stream);
    cudautils::memsetAsync(device, 0, N, stream);
    cudautils::copyAsync(host, device, N, stream);
    stream.synchronize();

    for(int i=0; i < N; ++i) {
      CHECK(host[i] == 0);
    }
  }

  //Fake the end-of-job signal.
  ar.postEndJobSignal_();
}

