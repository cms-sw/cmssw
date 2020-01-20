#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "catch.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

namespace {
  CUDAService makeCUDAService(edm::ParameterSet ps) {
    auto desc = edm::ConfigurationDescriptions("Service", "CUDAService");
    CUDAService::fillDescriptions(desc);
    desc.validate(ps, "CUDAService");
    return CUDAService(ps);
  }
}  // namespace

TEST_CASE("Tests of CUDAService", "[CUDAService]") {
  // Test setup: check if a simple CUDA runtime API call fails:
  // if so, skip the test with the CUDAService enabled
  int deviceCount = 0;
  auto ret = cudaGetDeviceCount(&deviceCount);

  if (ret != cudaSuccess) {
    WARN("Unable to query the CUDA capable devices from the CUDA runtime API: ("
         << ret << ") " << cudaGetErrorString(ret) << ". Running only tests not requiring devices.");
  }

  SECTION("CUDAService enabled") {
    edm::ParameterSet ps;
    ps.addUntrackedParameter("enabled", true);
    SECTION("Enabled only if there are CUDA capable GPUs") {
      auto cs = makeCUDAService(ps);
      if (deviceCount <= 0) {
        REQUIRE(cs.enabled() == false);
        WARN("CUDAService is disabled as there are no CUDA GPU devices");
      } else {
        REQUIRE(cs.enabled() == true);
        INFO("CUDAService is enabled");
      }
    }

    if (deviceCount <= 0) {
      return;
    }

    auto cs = makeCUDAService(ps);

    SECTION("CUDA Queries") {
      int driverVersion = 0, runtimeVersion = 0;
      ret = cudaDriverGetVersion(&driverVersion);
      if (ret != cudaSuccess) {
        FAIL("Unable to query the CUDA driver version from the CUDA runtime API: (" << ret << ") "
                                                                                    << cudaGetErrorString(ret));
      }
      ret = cudaRuntimeGetVersion(&runtimeVersion);
      if (ret != cudaSuccess) {
        FAIL("Unable to query the CUDA runtime API version: (" << ret << ") " << cudaGetErrorString(ret));
      }

      WARN("CUDA Driver Version / Runtime Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10
                                                     << " / " << runtimeVersion / 1000 << "."
                                                     << (runtimeVersion % 100) / 10);

      // Test that the number of devices found by the service
      // is the same as detected by the CUDA runtime API
      REQUIRE(cs.numberOfDevices() == deviceCount);
      WARN("Detected " << cs.numberOfDevices() << " CUDA Capable device(s)");

      // Test that the compute capabilities of each device
      // are the same as detected by the CUDA runtime API
      for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        ret = cudaGetDeviceProperties(&deviceProp, i);
        if (ret != cudaSuccess) {
          FAIL("Unable to query the CUDA properties for device " << i << " from the CUDA runtime API: (" << ret << ") "
                                                                 << cudaGetErrorString(ret));
        }

        REQUIRE(deviceProp.major == cs.computeCapability(i).first);
        REQUIRE(deviceProp.minor == cs.computeCapability(i).second);
        INFO("Device " << i << ": " << deviceProp.name << "\n CUDA Capability Major/Minor version number: "
                       << deviceProp.major << "." << deviceProp.minor);
      }
    }

    SECTION("CUDAService device free memory") {
      size_t mem = 0;
      int dev = -1;
      for (int i = 0; i < deviceCount; ++i) {
        size_t free, tot;
        cudaSetDevice(i);
        cudaMemGetInfo(&free, &tot);
        WARN("Device " << i << " memory total " << tot << " free " << free);
        if (free > mem) {
          mem = free;
          dev = i;
        }
      }
      WARN("Device with most free memory " << dev << "\n"
                                           << "     as given by CUDAService " << cs.deviceWithMostFreeMemory());
    }
  }

  SECTION("Force to be disabled") {
    edm::ParameterSet ps;
    ps.addUntrackedParameter("enabled", false);
    auto cs = makeCUDAService(ps);
    REQUIRE(cs.enabled() == false);
    REQUIRE(cs.numberOfDevices() == 0);
  }
}
