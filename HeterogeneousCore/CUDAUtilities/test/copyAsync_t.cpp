#include "catch.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

namespace {
  CUDAService makeCUDAService(edm::ParameterSet ps, edm::ActivityRegistry& ar) {
    auto desc = edm::ConfigurationDescriptions("Service", "CUDAService");
    CUDAService::fillDescriptions(desc);
    desc.validate(ps, "CUDAService");
    return CUDAService(ps, ar);
  }
}

TEST_CASE("copyAsync", "[cudaMemTools]") {
  exitSansCUDADevices();

  edm::ActivityRegistry ar;
  edm::ParameterSet ps;
  auto cs = makeCUDAService(ps, ar);

  auto current_device = cuda::device::current::get();
  auto stream = current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);

  SECTION("Host to device") {
    SECTION("Single element") {
      auto host_orig = cs.make_host_unique<int>(stream);
      *host_orig = 42;

      auto device = cs.make_device_unique<int>(stream);
      auto host = cs.make_host_unique<int>(stream);

      cudautils::copyAsync(device, host_orig, stream);
      cuda::memory::async::copy(host.get(), device.get(), sizeof(int), stream.id());
      stream.synchronize();

      REQUIRE(*host == 42);
    }

    SECTION("Multiple elements") {
      constexpr int N = 100;

      auto host_orig = cs.make_host_unique<int[]>(N, stream);
      for(int i=0; i<N; ++i) {
        host_orig[i] = i;
      }

      auto device = cs.make_device_unique<int[]>(N, stream);
      auto host = cs.make_host_unique<int[]>(N, stream);

      SECTION("Copy all") {
        cudautils::copyAsync(device, host_orig, N, stream);
        cuda::memory::async::copy(host.get(), device.get(), N*sizeof(int), stream.id());
        stream.synchronize();
        for(int i=0; i<N; ++i) {
          CHECK(host[i] == i);
        }
      }

      for(int i=0; i<N; ++i) {
        host_orig[i] = 200+i;
      }

      SECTION("Copy some") {
        cudautils::copyAsync(device, host_orig, 42, stream);
        cuda::memory::async::copy(host.get(), device.get(), 42*sizeof(int), stream.id());
        stream.synchronize();
        for(int i=0; i<42; ++i) {
          CHECK(host[i] == 200+i);
        }
      }
    }
  }

  SECTION("Device to host") {
    SECTION("Single element") {
      auto host_orig = cs.make_host_unique<int>(stream);
      *host_orig = 42;

      auto device = cs.make_device_unique<int>(stream);
      auto host = cs.make_host_unique<int>(stream);

      cuda::memory::async::copy(device.get(), host_orig.get(), sizeof(int), stream.id());
      cudautils::copyAsync(host, device, stream);
      stream.synchronize();

      REQUIRE(*host == 42);
    }

    SECTION("Multiple elements") {
      constexpr int N = 100;

      auto host_orig = cs.make_host_unique<int[]>(N, stream);
      for(int i=0; i<N; ++i) {
        host_orig[i] = i;
      }

      auto device = cs.make_device_unique<int[]>(N, stream);
      auto host = cs.make_host_unique<int[]>(N, stream);

      SECTION("Copy all") {
        cuda::memory::async::copy(device.get(), host_orig.get(), N*sizeof(int), stream.id());
        cudautils::copyAsync(host, device, N, stream);
        stream.synchronize();
        for(int i=0; i<N; ++i) {
          CHECK(host[i] == i);
        }
      }

      for(int i=0; i<N; ++i) {
        host_orig[i] = 200+i;
      }

      SECTION("Copy some") {
        cuda::memory::async::copy(device.get(), host_orig.get(), 42*sizeof(int), stream.id());
        cudautils::copyAsync(host, device, 42, stream);
        stream.synchronize();
        for(int i=0; i<42; ++i) {
          CHECK(host[i] == 200+i);
        }
      }
    }
  }

  //Fake the end-of-job signal.
  ar.postEndJobSignal_();
}

