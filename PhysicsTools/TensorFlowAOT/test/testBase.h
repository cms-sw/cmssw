/*
 * Base class for tests.
 */

#ifndef PHYSICSTOOLS_TENSORFLOWAOT_TEST_TESTBASE_H
#define PHYSICSTOOLS_TENSORFLOWAOT_TEST_TESTBASE_H

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
  void runCmd(const std::string& cmd);

  virtual void test() = 0;
};

void testBase::setUp() {
  dataPath_ =
      cmsswPath("/test/" + std::string(std::getenv("SCRAM_ARCH")) + "/" + boost::filesystem::unique_path().string());
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

void testBase::runCmd(const std::string& cmd) {
  // popen
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);

  // catch errors
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }

  // print the result
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != NULL) {
      result += buffer.data();
    }
  }
  std::cout << std::endl << result << std::endl;
}

#endif  // PHYSICSTOOLS_TENSORFLOWAOT_TEST_TESTBASE_H
