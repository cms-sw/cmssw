#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include <fmt/core.h>

#include <catch.hpp>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ResourceInformation.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"

namespace {
  std::string makeProcess(std::string const& name) {
    return fmt::format(R"_(
import FWCore.ParameterSet.Config as cms
process = cms.Process('{}')
)_",
                       name);
  }

  void addResourceInformationService(std::string& config) {
    config += R"_(
process.add_(cms.Service('ResourceInformationService'))
  )_";
  }

  void addCUDAService(std::string& config, bool enabled = true) {
    config += fmt::format(R"_(
process.add_(cms.Service('CUDAService',
  enabled = cms.untracked.bool({}),
  verbose = cms.untracked.bool(True)
))
  )_",
                          enabled ? "True" : "False");
  }

  edm::ServiceToken getServiceToken(std::string const& config) {
    std::unique_ptr<edm::ParameterSet> params;
    edm::makeParameterSets(config, params);
    return edm::ServiceToken(edm::ServiceRegistry::createServicesFromConfig(std::move(params)));
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

  std::string config = makeProcess("Test");
  addCUDAService(config);
  auto serviceToken = getServiceToken(config);
  edm::ServiceRegistry::Operate operate(serviceToken);

  SECTION("Enable the CUDAService only if there are CUDA capable GPUs") {
    edm::Service<CUDAInterface> cuda;
    if (deviceCount <= 0) {
      REQUIRE((not cuda or not cuda->enabled()));
      WARN("CUDAService is not present, or disabled because there are no CUDA GPU devices");
      return;
    } else {
      REQUIRE(cuda);
      REQUIRE(cuda->enabled());
      INFO("CUDAService is enabled");
    }
  }

  SECTION("CUDAService enabled") {
    int driverVersion = 0, runtimeVersion = 0;
    edm::Service<CUDAInterface> cuda;
    ret = cudaDriverGetVersion(&driverVersion);
    if (ret != cudaSuccess) {
      FAIL("Unable to query the CUDA driver version from the CUDA runtime API: (" << ret << ") "
                                                                                  << cudaGetErrorString(ret));
    }
    ret = cudaRuntimeGetVersion(&runtimeVersion);
    if (ret != cudaSuccess) {
      FAIL("Unable to query the CUDA runtime API version: (" << ret << ") " << cudaGetErrorString(ret));
    }

    SECTION("CUDA Queries") {
      WARN("CUDA Driver Version / Runtime Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10
                                                     << " / " << runtimeVersion / 1000 << "."
                                                     << (runtimeVersion % 100) / 10);

      // Test that the number of devices found by the service
      // is the same as detected by the CUDA runtime API
      REQUIRE(cuda->numberOfDevices() == deviceCount);
      WARN("Detected " << cuda->numberOfDevices() << " CUDA Capable device(s)");

      // Test that the compute capabilities of each device
      // are the same as detected by the CUDA runtime API
      for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        ret = cudaGetDeviceProperties(&deviceProp, i);
        if (ret != cudaSuccess) {
          FAIL("Unable to query the CUDA properties for device " << i << " from the CUDA runtime API: (" << ret << ") "
                                                                 << cudaGetErrorString(ret));
        }

        REQUIRE(deviceProp.major == cuda->computeCapability(i).first);
        REQUIRE(deviceProp.minor == cuda->computeCapability(i).second);
        INFO("Device " << i << ": " << deviceProp.name << "\n CUDA Capability Major/Minor version number: "
                       << deviceProp.major << "." << deviceProp.minor);
      }
    }

    SECTION("With ResourceInformationService available") {
      std::string config = makeProcess("Test");
      addResourceInformationService(config);
      addCUDAService(config);
      auto serviceToken = getServiceToken(config);
      edm::ServiceRegistry::Operate operate(serviceToken);

      edm::Service<CUDAInterface> cuda;
      REQUIRE(cuda);
      REQUIRE(cuda->enabled());
      edm::Service<edm::ResourceInformation> ri;
      REQUIRE(ri);
      REQUIRE(ri->gpuModels().size() > 0);
      REQUIRE(ri->nvidiaDriverVersion().size() > 0);
      REQUIRE(ri->cudaDriverVersion() == driverVersion);
      REQUIRE(ri->cudaRuntimeVersion() == runtimeVersion);
    }
  }

  SECTION("CUDAService disabled") {
    std::string config = makeProcess("Test");
    addCUDAService(config, false);
    auto serviceToken = getServiceToken(config);
    edm::ServiceRegistry::Operate operate(serviceToken);

    edm::Service<CUDAInterface> cuda;
    REQUIRE(cuda->enabled() == false);
    REQUIRE(cuda->numberOfDevices() == 0);
  }
}
