/*
 * Base class for tests.
 *
 */

#ifndef PHYSICSTOOLS_PYTORCH_TEST_TESTBASE_H
#define PHYSICSTOOLS_PYTORCH_TEST_TESTBASE_H

#include <boost/filesystem.hpp>
#include <filesystem>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

class testBasePyTorch : public CppUnit::TestFixture {
public:
  std::string dataPath_;
  std::string testPath_;

  void setUp();
  void tearDown();
  std::string cmsswPath(std::string path);

  virtual void test() = 0;

};

void testBasePyTorch::setUp() {
  dataPath_ =
      cmsswPath("/test/" + std::string(std::getenv("SCRAM_ARCH")) + "/" + boost::filesystem::unique_path().string());

  // create the graph
  testPath_ = cmsswPath("/src/PhysicsTools/PyTorch/test");
}

void testBasePyTorch::tearDown() {
  if (std::filesystem::exists(dataPath_)) {
    std::filesystem::remove_all(dataPath_);
  }
}

std::string testBasePyTorch::cmsswPath(std::string path) {
  if (path.size() > 0 && path.substr(0, 1) != "/") {
    path = "/" + path;
  }

  std::string base = std::string(std::getenv("CMSSW_BASE"));
  std::string releaseBase = std::string(std::getenv("CMSSW_RELEASE_BASE"));

  return (std::filesystem::exists(base.c_str()) ? base : releaseBase) + path;
}

#endif  // PHYSICSTOOLS_PYTORCH_TEST_TESTBASE_H
