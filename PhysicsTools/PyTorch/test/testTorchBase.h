#ifndef PhysicsTools_PyTorch_test_testTorchBase_h
#define PhysicsTools_PyTorch_test_testTorchBase_h

#include <filesystem>
#include <memory>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"

namespace torchtest {

  class testTorchBase : public CppUnit::TestFixture {
  public:
    void setUp() {
      auto path = "/test/" + std::string(std::getenv("SCRAM_ARCH")) + "/" + boost::filesystem::unique_path().string();
      model_path_ = torchtest::cmsswPath(path);

      auto test_path = torchtest::cmsswPath("/src/PhysicsTools/PyTorch/test");
      std::string cmd = buildCmd();
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
    }

    void tearDown() {
      if (std::filesystem::exists(model_path_)) {
        std::filesystem::remove_all(model_path_);
      }
    }

    virtual std::string script() const = 0;

    std::string modelPath() const { return model_path_; }

  private:
    std::string model_path_;
    std::string image_ = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.11";
    std::string python_exe_ = "python";
    std::string apptainer_cmd_ = "apptainer exec -B";
    std::string test_path_ = torchtest::cmsswPath("/src/PhysicsTools/PyTorch/test") + "/";

    const std::string buildCmd() const {
      return apptainer_cmd_ + " " + torchtest::cmsswPath("") + " " + image_ + " " + python_exe_ + " " + test_path_ +
             script() + " " + model_path_;
    }
  };

}  // namespace torchtest

#endif  // PhysicsTools_PyTorch_test_testTorchBase_h