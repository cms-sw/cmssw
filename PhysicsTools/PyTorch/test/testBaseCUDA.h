/*
 * Base class for tests.
 *
 */

#ifndef PHYSICSTOOLS_PYTORCH_TEST_TESTBASECUDA_H
#define PHYSICSTOOLS_PYTORCH_TEST_TESTBASECUDA_H

#include <boost/filesystem.hpp>
#include <filesystem>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

#include "FWCore/AbstractServices/interface/ResourceInformation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Exception.h"

class testBasePyTorchCUDA : public CppUnit::TestFixture {
public:
  std::string dataPath_;

  void setUp();
  void tearDown();
  std::string cmsswPath(std::string path);

  virtual std::string pyScript() const = 0;

  virtual void test() = 0;
};

void testBasePyTorchCUDA::setUp() {
  dataPath_ =
      cmsswPath("/test/" + std::string(std::getenv("SCRAM_ARCH")) + "/" + boost::filesystem::unique_path().string());

  // create the graph using apptainer
  std::string testPath = cmsswPath("/src/PhysicsTools/PyTorch/test");
  std::string cmd = "apptainer exec -B " + cmsswPath("") +
                    "  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.11  python " + testPath + "/" +
                    pyScript() + " " + dataPath_;
  std::cout << "cmd: " << cmd << std::endl;
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("Failed to run apptainer to prepare the PyTorch test model: " + cmd);
  }
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != NULL) {
      result += buffer.data();
    }
  }
  std::cout << std::endl << result << std::endl;
}
void testBasePyTorchCUDA::tearDown() {
  if (std::filesystem::exists(dataPath_)) {
    std::filesystem::remove_all(dataPath_);
  }
}

std::string testBasePyTorchCUDA::cmsswPath(std::string path) {
  if (path.size() > 0 && path.substr(0, 1) != "/") {
    path = "/" + path;
  }

  std::string base = std::string(std::getenv("CMSSW_BASE"));
  std::string releaseBase = std::string(std::getenv("CMSSW_RELEASE_BASE"));

  return (std::filesystem::exists(base.c_str()) ? base : releaseBase) + path;
}

#endif  // PHYSICSTOOLS_PYTORCH_TEST_TESTBASE_H
