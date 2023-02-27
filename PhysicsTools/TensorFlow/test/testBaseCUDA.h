/*
 * Base class for tests.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_TEST_TESTBASECUDA_H
#define PHYSICSTOOLS_TENSORFLOW_TEST_TESTBASECUDA_H

#include <boost/filesystem.hpp>
#include <filesystem>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>
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
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

namespace {
  CUDAService makeCUDAService(edm::ParameterSet ps) {
    auto desc = edm::ConfigurationDescriptions("Service", "CUDAService");
    CUDAService::fillDescriptions(desc);
    desc.validate(ps, "CUDAService");
    return CUDAService(ps);
  }
}  // namespace

class testBaseCUDA : public CppUnit::TestFixture {
public:
  std::string dataPath_;

  void setUp();
  void tearDown();
  std::string cmsswPath(std::string path);

  virtual void test() = 0;

  virtual std::string pyScript() const = 0;
};

void testBaseCUDA::setUp() {
  dataPath_ =
      cmsswPath("/test/" + std::string(std::getenv("SCRAM_ARCH")) + "/" + boost::filesystem::unique_path().string());

  // create the graph
  std::string testPath = cmsswPath("/src/PhysicsTools/TensorFlow/test");
  std::string cmd = "python3 -W ignore " + testPath + "/" + pyScript() + " " + dataPath_;
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != NULL) {
      result += buffer.data();
    }
  }
  std::cout << std::endl << result << std::endl;
}

void testBaseCUDA::tearDown() {
  if (std::filesystem::exists(dataPath_)) {
    std::filesystem::remove_all(dataPath_);
  }
}

std::string testBaseCUDA::cmsswPath(std::string path) {
  if (path.size() > 0 && path.substr(0, 1) != "/") {
    path = "/" + path;
  }

  std::string base = std::string(std::getenv("CMSSW_BASE"));
  std::string releaseBase = std::string(std::getenv("CMSSW_RELEASE_BASE"));

  return (std::filesystem::exists(base.c_str()) ? base : releaseBase) + path;
}

#endif  // PHYSICSTOOLS_TENSORFLOW_TEST_TESTBASECUDA_H
