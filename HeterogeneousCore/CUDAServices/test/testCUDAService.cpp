#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "catch.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ResourceInformation.h"
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

  // Make Service system available as CUDAService depends on ResourceInformationService
  std::vector<edm::ParameterSet> psets;
  edm::ServiceToken serviceToken = edm::ServiceRegistry::createSet(psets);
  edm::ServiceRegistry::Operate operate(serviceToken);

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

    SECTION("CUDA Queries") {
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

      auto cs = makeCUDAService(edm::ParameterSet{});
      REQUIRE(cs.enabled());
      edm::Service<edm::ResourceInformation> ri;
      REQUIRE(ri->gpuModels().size() > 0);
      REQUIRE(ri->nvidiaDriverVersion().size() > 0);
      REQUIRE(ri->cudaDriverVersion() == driverVersion);
      REQUIRE(ri->cudaRuntimeVersion() == runtimeVersion);
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
