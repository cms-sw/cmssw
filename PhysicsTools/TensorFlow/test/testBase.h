/*
 * Base class for tests.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_TEST_TESTBASE_H
#define PHYSICSTOOLS_TENSORFLOW_TEST_TESTBASE_H

#include <boost/filesystem.hpp>
#include <filesystem>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

class testBase : public CppUnit::TestFixture {
public:
  std::string dataPath_;

  void setUp();
  void tearDown();
  std::string cmsswPath(std::string path);

  virtual void checkAll() = 0;
  virtual std::string pyScript() const = 0;
};

void testBase::setUp() {
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

void testBase::tearDown() {
  if (std::filesystem::exists(dataPath_)) {
    std::filesystem::remove_all(dataPath_);
  }
}

std::string testBase::cmsswPath(std::string path) {
  if (path.size() > 0 && path.substr(0, 1) != "/") {
    path = "/" + path;
  }

  std::string base = std::string(std::getenv("CMSSW_BASE"));
  std::string releaseBase = std::string(std::getenv("CMSSW_RELEASE_BASE"));

  return (std::filesystem::exists(base.c_str()) ? base : releaseBase) + path;
}

#endif  // PHYSICSTOOLS_TENSORFLOW_TEST_TESTBASE_H
