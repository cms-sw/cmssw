#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include <hip/hip_runtime.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ResourceInformation.h"
#include "HeterogeneousCore/ROCmServices/interface/ROCmService.h"
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"

namespace {
  ROCmService makeROCmService(edm::ParameterSet ps) {
    auto desc = edm::ConfigurationDescriptions("Service", "ROCmService");
    ROCmService::fillDescriptions(desc);
    desc.validate(ps, "ROCmService");
    return ROCmService(ps);
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

  // Make Service system available as ROCmService depends on ResourceInformationService
  std::vector<edm::ParameterSet> psets;
  edm::ServiceToken serviceToken = edm::ServiceRegistry::createSet(psets);
  edm::ServiceRegistry::Operate operate(serviceToken);

  SECTION("ROCmService enabled") {
    edm::ParameterSet ps;
    ps.addUntrackedParameter("enabled", true);
    SECTION("Enabled only if there are ROCm capable GPUs") {
      auto cs = makeROCmService(ps);
      if (deviceCount <= 0) {
        REQUIRE(cs.enabled() == false);
        WARN("ROCmService is disabled as there are no ROCm GPU devices");
      } else {
        REQUIRE(cs.enabled() == true);
        INFO("ROCmService is enabled");
      }
    }

    if (deviceCount <= 0) {
      return;
    }

    auto cs = makeROCmService(ps);
    int driverVersion = 0, runtimeVersion = 0;
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
      REQUIRE(cs.numberOfDevices() == deviceCount);
      WARN("Detected " << cs.numberOfDevices() << " ROCm Capable device(s)");

      // Test that the compute capabilities of each device
      // are the same as detected by the ROCm runtime API
      for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t deviceProp;
        ret = hipGetDeviceProperties(&deviceProp, i);
        if (ret != hipSuccess) {
          FAIL("Unable to query the ROCm properties for device " << i << " from the ROCm runtime API: (" << ret << ") "
                                                                 << hipGetErrorString(ret));
        }

        REQUIRE(deviceProp.major == cs.computeCapability(i).first);
        REQUIRE(deviceProp.minor == cs.computeCapability(i).second);
        INFO("Device " << i << ": " << deviceProp.name << "\n ROCm Capability Major/Minor version number: "
                       << deviceProp.major << "." << deviceProp.minor);
      }
    }

    SECTION("ROCmService device free memory") {
      size_t mem = 0;
      int dev = -1;
      for (int i = 0; i < deviceCount; ++i) {
        size_t free, tot;
        REQUIRE_NOTHROW(hipCheck(hipSetDevice(i)));
        REQUIRE_NOTHROW(hipCheck(hipMemGetInfo(&free, &tot)));
        WARN("Device " << i << " memory total " << tot << " free " << free);
        if (free > mem) {
          mem = free;
          dev = i;
        }
      }
      WARN("Device with most free memory " << dev << "\n"
                                           << "     as given by ROCmService " << cs.deviceWithMostFreeMemory());
    }

    SECTION("With ResourceInformationService available") {
      edmplugin::PluginManager::configure(edmplugin::standard::config());

      std::string const config = R"_(import FWCore.ParameterSet.Config as cms
process = cms.Process('Test')
process.add_(cms.Service('ResourceInformationService'))
)_";
      std::unique_ptr<edm::ParameterSet> params;
      edm::makeParameterSets(config, params);
      edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(std::move(params)));
      edm::ServiceRegistry::Operate operate2(tempToken);

      auto cs = makeROCmService(edm::ParameterSet{});
      REQUIRE(cs.enabled());
      edm::Service<edm::ResourceInformation> ri;
      REQUIRE(ri->gpuModels().size() > 0);
      /*
      REQUIRE(ri->nvidiaDriverVersion().size() > 0);
      REQUIRE(ri->cudaDriverVersion() == driverVersion);
      REQUIRE(ri->cudaRuntimeVersion() == runtimeVersion);
      */
    }
  }

  SECTION("Force to be disabled") {
    edm::ParameterSet ps;
    ps.addUntrackedParameter("enabled", false);
    auto cs = makeROCmService(ps);
    REQUIRE(cs.enabled() == false);
    REQUIRE(cs.numberOfDevices() == 0);
  }
}
