#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include <hip/hip_runtime.h>

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
#include "HeterogeneousCore/ROCmServices/interface/ROCmInterface.h"

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

  void addROCmService(std::string& config, bool enabled = true) {
    config += fmt::format(R"_(
process.add_(cms.Service('ROCmService',
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

TEST_CASE("Tests of ROCmService", "[ROCmService]") {
  // Test setup: check if a simple ROCm runtime API call fails:
  // if so, skip the test with the ROCmService enabled
  int deviceCount = 0;
  auto ret = hipGetDeviceCount(&deviceCount);

  if (ret != hipSuccess) {
    WARN("Unable to query the ROCm capable devices from the ROCm runtime API: ("
         << ret << ") " << hipGetErrorString(ret) << ". Running only tests not requiring devices.");
  }

  std::string config = makeProcess("Test");
  addROCmService(config);
  auto serviceToken = getServiceToken(config);
  edm::ServiceRegistry::Operate operate(serviceToken);

  SECTION("Enable the ROCmService only if there are ROCm capable GPUs") {
    edm::Service<ROCmInterface> service;
    if (deviceCount <= 0) {
      REQUIRE((not service or not service->enabled()));
      WARN("ROCmService is not present, or disabled because there are no ROCm GPU devices");
      return;
    } else {
      REQUIRE(service);
      REQUIRE(service->enabled());
      INFO("ROCmService is enabled");
    }
  }

  SECTION("ROCmService enabled") {
    int driverVersion = 0, runtimeVersion = 0;
    edm::Service<ROCmInterface> service;
    ret = hipDriverGetVersion(&driverVersion);
    if (ret != hipSuccess) {
      FAIL("Unable to query the ROCm driver version from the ROCm runtime API: (" << ret << ") "
                                                                                  << hipGetErrorString(ret));
    }
    ret = hipRuntimeGetVersion(&runtimeVersion);
    if (ret != hipSuccess) {
      FAIL("Unable to query the ROCm runtime API version: (" << ret << ") " << hipGetErrorString(ret));
    }

    SECTION("ROCm Queries") {
      WARN("ROCm Driver Version / Runtime Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10
                                                     << " / " << runtimeVersion / 1000 << "."
                                                     << (runtimeVersion % 100) / 10);

      // Test that the number of devices found by the service
      // is the same as detected by the ROCm runtime API
      REQUIRE(service->numberOfDevices() == deviceCount);
      WARN("Detected " << service->numberOfDevices() << " ROCm Capable device(s)");

      // Test that the compute capabilities of each device
      // are the same as detected by the ROCm runtime API
      for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t deviceProp;
        ret = hipGetDeviceProperties(&deviceProp, i);
        if (ret != hipSuccess) {
          FAIL("Unable to query the ROCm properties for device " << i << " from the ROCm runtime API: (" << ret << ") "
                                                                 << hipGetErrorString(ret));
        }

        REQUIRE(deviceProp.major == service->computeCapability(i).first);
        REQUIRE(deviceProp.minor == service->computeCapability(i).second);
        INFO("Device " << i << ": " << deviceProp.name << "\n ROCm Capability Major/Minor version number: "
                       << deviceProp.major << "." << deviceProp.minor);
      }
    }

    SECTION("With ResourceInformationService available") {
      std::string config = makeProcess("Test");
      addResourceInformationService(config);
      addROCmService(config);
      auto serviceToken = getServiceToken(config);
      edm::ServiceRegistry::Operate operate(serviceToken);

      edm::Service<ROCmInterface> service;
      REQUIRE(service);
      REQUIRE(service->enabled());
      edm::Service<edm::ResourceInformation> ri;
      REQUIRE(ri->gpuModels().size() > 0);
      /*
      REQUIRE(ri->amdDriverVersion().size() > 0);
      REQUIRE(ri->rocmDriverVersion() == driverVersion);
      REQUIRE(ri->rocmRuntimeVersion() == runtimeVersion);
      */
    }
  }

  SECTION("Force to be disabled") {
    std::string config = makeProcess("Test");
    addROCmService(config, false);
    auto serviceToken = getServiceToken(config);
    edm::ServiceRegistry::Operate operate(serviceToken);

    edm::Service<ROCmInterface> service;
    REQUIRE(service->enabled() == false);
    REQUIRE(service->numberOfDevices() == 0);
  }
}
